import warnings
warnings.filterwarnings('ignore')

import os
from datetime import datetime
import copick
from tqdm import tqdm
import numpy as np
from glob import glob
import time
import cc3d
import pandas as pd

from monai.networks.nets import UNet, DynUNet
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    Orientationd,  
    AsDiscrete,  
    RandFlipd, 
    RandRotate90d, 
    NormalizeIntensityd,
    RandCropByLabelClassesd,
    Rotate90, 
)

from src.train.trainer import *
from src.train.dataloader import gen_train_val_dataloader
from src.train.loss import FbetaLoss
from src.utils.helper import *
from src.utils.dataset import *
from tools.search_best_prob_thresh import search_best_prob_thresh

def get_dataset_df(cfg) -> None:
    root = copick.from_file('./working/copick.config')
    run_names = [r.name for r in root.runs]
    num_folds = len(run_names)
    df = []
    if cfg.mode=='local':
        for i in range(num_folds):
            test_name = run_names[i]
            val_name = run_names[i+1 if i<num_folds-1 else 0] # shift one
            train_names = run_names.copy()
            train_names.remove(test_name)
            train_names.remove(val_name)
            df.append({'train':train_names, 'val':val_name, 'test':test_name})

    if cfg.mode=='sub':
        for i in range(num_folds):
            val_name = run_names[i]
            train_names = run_names.copy()
            train_names.remove(val_name)
            df.append({'train':train_names, 'val':val_name})
    
    cfg.data_split = pd.DataFrame(df)

def run_train(cfg, stage):
    seed_everything(cfg.seed)
    
    cfg.stage = stage
    print(f'\n ----- STAGE {cfg.stage} -----\n')

    # setting copick configs
    root = copick.from_file('./working/copick.config')
    copick_user_name = 'copickUtils'
    copick_segmentation_name = 'paintedPicks'
    voxel_size = 10
    tomo_types = ['ctfdeconvolved', 'isonetcorrected', 'wbp', 'denoised']

    if stage == 0:
        tomo_type = tomo_types[:3] # 'ctfdeconvolved', 'isonetcorrected', 'wbp'
    elif stage == 1:
        tomo_type = tomo_types[3:] # 'denoised'
    
    # getting tomograms and their segmentation mask arrays
    train_files, val_files = [], []
    train_run_names = cfg['data_split'].iloc[cfg.fold]['train']
    val_run_names = cfg['data_split'].iloc[cfg.fold]['val']
    for run in tqdm(root.runs):
        for tt in tomo_type:
            tomo = run.get_voxel_spacing(voxel_size).get_tomogram(tt).numpy()
            seg = run.get_segmentations(
                name=copick_segmentation_name, user_id=copick_user_name, voxel_size=voxel_size, is_multilabel=True
                )[0].numpy()
            
            if run.name in train_run_names: 
                train_files.append({"image": tomo, "label": seg})
            elif run.name in val_run_names: 
                val_files.append({"image": tomo, "label": seg})

    train_loader, val_loader = gen_train_val_dataloader(train_files, val_files, cfg) # apply transforms

    print(f'Train: {len(train_files)} / Val: {len(val_files)}')
    print(f'Device: {cfg.device}')
    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES,
        channels=(48, 64, 80, 80), strides=(2, 2, 1), num_res_units=1,
    ).to(cfg.device)

    if stage == 1:
        path_to_model = sorted(glob(f'./working/train/{cfg.model_folder}/*_{cfg.fold}.pth'))[0]
        model.load_state_dict(torch.load(path_to_model))
        for param in model.parameters():
            param.requires_grad = True

    lr = float(cfg.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr)  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(cfg.epochs*0.3), round(cfg.epochs*0.8)], gamma=0.1)
    metric_func = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
    loss_func = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  # `softmax=True` for multiclass

    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)
    train(
        cfg, train_loader, val_loader, 
        model, loss_func, metric_func, optimizer, scheduler,
        post_pred, post_label,
        )

def do_cv(cfg):
    if not cfg.do_cv:
        return None
    model_paths = sorted(glob(f'./working/train/{cfg.model_folder}/*.pth'))
    root = copick.from_file('./working/copick.config')
    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES,
        channels=(48, 64, 80, 80), strides=(2, 2, 1), num_res_units=1,
    ).to(cfg.device)
    inference_transforms = Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS")
    ])

    weight = get_gaussian_weight(
        cfg.patch_size, cfg.patch_size[1]//2-0, 0
        ).to(cfg.device)
    with torch.no_grad():
        run = root.runs[cfg.fold]
        # pick up the trained model which corresponds to the current fold
        for path in model_paths:
            if cfg.fold == int(path[-len('.pth')-1]):
                print(f'Model: {os.path.basename(path)}')
                model.load_state_dict(torch.load(path))
                model.eval()

        print(f'** TEST {run.name} FOR CV **')
        start = time.time()

        tomo = run.get_voxel_spacing(10)
        tomo = tomo.get_tomogram('denoised').numpy()
        original_shape = tomo.shape
        tomo_patches, coordinates  = extract_3d_patches_minimal_overlap([tomo], cfg.patch_size, cfg.overlap)
        tomo_patched_data = [{"image": img} for img in tomo_patches]
        tomo_ds = CacheDataset(data=tomo_patched_data, transform=inference_transforms, cache_rate=1.0)

        reconstructed = torch.zeros(
            [NUM_CLASSES, original_shape[0], original_shape[1], original_shape[2]]
            ).to('cuda')  # To track overlapping regions
        count = torch.zeros(
            [NUM_CLASSES, original_shape[0], original_shape[1], original_shape[2]]
            ).to('cuda')
        for i in range(len(tomo_ds)):
            if cfg.tta:
                # w/o rotate
                input_tensor = tomo_ds[i]['image'].unsqueeze(0).to('cuda')
                model_output_tmp = model(input_tensor)
                model_output_tmp = model_output_tmp.squeeze(0)
                model_outputs_tta = [model_output_tmp]
                # tta with rotate90(k=1~3)
                for k in range(1, cfg.tta_k_rotate+1):
                    input_tensor = tomo_ds[i]['image'].to("cuda")
                    rotate = Rotate90(k=k, spatial_axes=(0, 2))
                    rotate_inverse = Rotate90(k=4-k, spatial_axes=(0, 2))
                    input_tensor = rotate(input_tensor)
                    input_tensor = input_tensor.unsqueeze(0)
                    model_output_tmp = model(input_tensor)
                    model_output_tmp = model_output_tmp.squeeze(0)
                    model_output_tmp = rotate_inverse(model_output_tmp)
                    model_outputs_tta.append(model_output_tmp)
                model_output = torch.stack(model_outputs_tta, 0).mean(0)
                model_output = model_output.unsqueeze(0)
            else:
                input_tensor = tomo_ds[i]['image'].unsqueeze(0).to('cuda')
                model_output = model(input_tensor)

            prob = torch.softmax(model_output[0], dim=0) # (7, 96, 96, 96)

            reconstructed[
                :, 
                coordinates[i][0]:coordinates[i][0] + cfg.patch_size[0],
                coordinates[i][1]:coordinates[i][1] + cfg.patch_size[1],
                coordinates[i][2]:coordinates[i][2] + cfg.patch_size[2]
            ] += prob

            count[
                :, 
                coordinates[i][0]:coordinates[i][0] + cfg.patch_size[0],
                coordinates[i][1]:coordinates[i][1] + cfg.patch_size[1],
                coordinates[i][2]:coordinates[i][2] + cfg.patch_size[2]
            ] += weight

        reconstructed /= count
        
        if isinstance(cfg.prob_thresh, float):
            thresh_prob = reconstructed > cfg.prob_thresh
            _, max_classes = thresh_prob.max(dim=0)
            thresh_max_classes = max_classes

        else:
            max_probs, max_classes = torch.max(reconstructed, dim=0)
            thresh_prob = torch.zeros_like(reconstructed)
            thresh_max_classes = torch.zeros_like(reconstructed[0])
            for ch in range(NUM_CLASSES):
                max_channel_is_one = torch.where(max_classes==ch, 1, 0)
                thresh_prob[ch] = max_probs * max_channel_is_one > cfg.certainty_threshold[ch]
                thresh_prob[ch] = torch.where(thresh_prob[ch]==1, ch, 0)
                thresh_max_classes += thresh_prob[ch]

        thresh_max_classes = thresh_max_classes.cpu().numpy()
        
        location = {}
        for c in CLASSES:
            cc = cc3d.connected_components(thresh_max_classes == c)
            stats = cc3d.statistics(cc)
            zyx=stats['centroids'][1:]*10.012444 #https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895#3040071
            zyx_large = zyx[stats['voxel_counts'][1:] > cfg.blob_threshold]
            xyz =np.ascontiguousarray(zyx_large[:,::-1])
            location[ID_TO_NAME[c]] = xyz

        location_df = dict_to_df(location, run.name)
        inference_time = time.time() - start

    location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))
    # location_df.to_csv("submission.csv", index=False)

    #-- scoring
    gb, lb_score = compute_lb(location_df, f'{cfg.local_kaggle_dataset_dir}/train/overlay/ExperimentRuns')
    print(f'LB Score: {lb_score} / Inference Time: {inference_time}')

def make_output_folder(cfg) -> None:
    dt = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder_name = f'{dt}_{cfg.exp_name}'
    output_folder_name += f'{cfg.batch_size}_{cfg.epochs}_{"x".join([str(i) for i in cfg.patch_size])}'
    cfg.model_folder = output_folder_name
    os.makedirs(f'./working/train/{output_folder_name}', exist_ok=True)


if __name__ == '__main__':
    cfg = dotdict(load_config('config.yml'))
    _, _ = make_output_folder(cfg), get_dataset_df(cfg)

    # for i in [0,1,2,3,4,5,6]: 
    for i in [0]:
        cfg.fold = i
        print(
            f'\n ================================'
            f'\n         CZII TRAIN FOLD {i}'
            f'\n ================================'
            )
        run_train(cfg, stage=0) # pretrain w/ 'ctfdeconvolved', 'isonetcorrected', and 'wbp'
        run_train(cfg, stage=1) # train w/ 'denoised'
    
        cfg.prob_thresh = search_best_prob_thresh(cfg) if cfg.find_best_thresh else cfg.prob_thresh
        print(f'\nThreshold: {cfg.prob_thresh}\n')
        do_cv(cfg)
            