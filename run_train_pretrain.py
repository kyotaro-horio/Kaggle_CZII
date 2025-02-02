import warnings
warnings.filterwarnings('ignore')
import os
import shutil
from datetime import datetime
import copick
from tqdm import tqdm
import numpy as np
from glob import glob
# import copy
from pprint import pprint
import time
import cc3d
import pandas as pd
from scipy.optimize import linear_sum_assignment

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

from train.trainer import *
from train.dataloader import *
from train.loss import FbetaLoss
from czii_helper.helper import *
from czii_helper.helper2 import *
from czii_helper.dataset import *
from find_the_best_certainty_threshold import find_the_best_certainty_threshold

def generate_split_dataset():
    root = copick.from_file('./working/copick.config')
    names = [r.name for r in root.runs]
    n_fold = len(names)

    names_test = names.copy()
    names_val = names_test[1:] + names_test[:1]
    names_trn = []
    for i in range(n_fold):
        names_trn_tmp = names_test.copy()
        names_trn_tmp.remove(names_test[i])
        names_trn_tmp.remove(names_val[i])
        names_trn.append(names_trn_tmp)

    df = []
    for trn, val, test in zip(names_trn, names_val, names_test):
        df.append({
            'train':trn, 
            'validation':val,
            'test':test, 
        })
    return pd.DataFrame(df)

def generate_split_dataset_submit():
    root = copick.from_file('./working/copick.config')
    names = [r.name for r in root.runs]
    n_fold = len(names)

    names_val = names.copy()
    names_trn = []
    for i in range(n_fold):
        names_trn_tmp = names_val.copy()
        names_trn_tmp.remove(names_val[i])
        names_trn.append(names_trn_tmp)

    df = []
    for trn, val in zip(names_trn, names_val):
        df.append({
            'train':trn, 
            'validation':val,  
        })
    return pd.DataFrame(df)

def run_train(cfg, fold, stage):

    cfg.fold = fold
    cfg.stage = stage

    os.makedirs('./working',exist_ok=True)
    root = copick.from_file('./working/copick.config')

    copick_user_name = 'copickUtils'
    copick_segmentation_name = 'paintedPicks'
    voxel_size = 10
    tomo_type = ['ctfdeconvolved', 'isonetcorrected', 'wbp', 'denoised']

    # get tomograms and their segmentation mask arrays for each data split
    if stage == 1:
        print(f'\n[stage{stage}]\n')
        trn_files = []
        val_files = []
        for run in tqdm(root.runs):
            for tt in tomo_type[:3]:
                tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tt).numpy()
                segmentation = run.get_segmentations(name=copick_segmentation_name, user_id=copick_user_name, voxel_size=voxel_size, is_multilabel=True)[0].numpy()
                
                if run.name in cfg['data_split'].iloc[fold]['train']:
                    trn_files.append({"image": tomogram, "label": segmentation})
                elif run.name in cfg['data_split'].iloc[fold]['validation']:
                    val_files.append({"image": tomogram, "label": segmentation})

        trn_loader, val_loader = generate_trn_val_dataloader(trn_files, val_files, cfg)

    elif stage == 2:
        print(f'\n[stage{stage}]\n')
        trn_files = []
        val_files = []
        for run in tqdm(root.runs):
            tomogram = run.get_voxel_spacing(voxel_size).get_tomogram(tomo_type[3]).numpy() #tomo_type='denoise' only
            segmentation = run.get_segmentations(name=copick_segmentation_name, user_id=copick_user_name, voxel_size=voxel_size, is_multilabel=True)[0].numpy()
            
            if run.name in cfg['data_split'].iloc[fold]['train']:
                trn_files.append({"image": tomogram, "label": segmentation})
            elif run.name in cfg['data_split'].iloc[fold]['validation']:
                val_files.append({"image": tomogram, "label": segmentation})

        trn_loader, val_loader = generate_trn_val_dataloader(trn_files, val_files, cfg)
# --- 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    text = f'\ndataset: '
    text += f'\n\ttrain:      {len(trn_files)} samples'
    text += f'\n\tvalidation: {len(val_files)} samples'
    text += f'\ndevice:'  
    text += f'\n\t{device}'
    print(text)
        
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=n_classes,
        channels=(48, 64, 80, 80),
        strides=(2, 2, 1),
        num_res_units=1,
    ).to(device)

    if stage == 2:
        path_to_model = sorted(glob(f'./working/train/{cfg.model_folder}/*_{fold}.pth'))[0]
        model.load_state_dict(torch.load(path_to_model))
        for param in model.parameters():
            param.requires_grad = True

    lr = float(cfg.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr)  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(cfg.epochs*0.3), round(cfg.epochs*0.8)], gamma=0.1)
    dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  # softmax=True for multiclass
# ---

    post_pred = AsDiscrete(argmax=True, to_onehot=n_classes)
    post_label = AsDiscrete(to_onehot=n_classes)
    train(
        cfg, trn_loader, val_loader, 
        model, loss_function, dice_metric, optimizer, scheduler, device, 
        post_pred, post_label,
        )


def get_gaussian_weight(kernel_size, sigma=1, muu=0):
    x, y, z = torch.meshgrid(
        torch.linspace(-kernel_size//2, kernel_size//2, kernel_size),
        torch.linspace(-kernel_size//2, kernel_size//2, kernel_size), 
        torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        )
    mean = torch.Tensor([muu, muu, muu])  # Mean of the Gaussian
    std_dev = torch.Tensor([sigma, sigma, sigma])  # Standard deviation along each axis

    gaussian = torch.exp(
        -0.5 * (
            ((x - mean[0]) / std_dev[0])**2 +
            ((y - mean[1]) / std_dev[1])**2 +
            ((z - mean[2]) / std_dev[2])**2
        )
    )

    # gaussian = gaussian / gaussian.max()
    # gaussian = (gaussian*255).to(torch.int16)
    
    return gaussian


def do_one_eval(truth, predict, threshold):
    P=len(predict)
    T=len(truth)

    if P==0:
        hit=[[],[]]
        miss=np.arange(T).tolist()
        fp=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    if T==0:
        hit=[[],[]]
        fp=np.arange(P).tolist()
        miss=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    #---
    distance = predict.reshape(P,1,3)-truth.reshape(1,T,3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss,t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp,p_index)].tolist()

    metric = [P,T,len(hit[0]),len(miss),len(fp)] #for lb metric F-beta copmutation
    return hit, fp, miss, metric


def compute_lb(submit_df, overlay_dir):

    valid_id = list(submit_df['experiment'].unique())
    print(valid_id)

    eval_df = []
    for id in valid_id:
        truth = read_one_truth(id, overlay_dir) #=f'{valid_dir}/overlay/ExperimentRuns')
        id_df = submit_df[submit_df['experiment'] == id]
        
        for p in PARTICLE:
            p = dotdict(p)            
            xyz_truth = truth[p.name]
            xyz_predict = id_df[id_df['particle_type'] == p.name][['x', 'y', 'z']].values
            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius* 0.5)
            eval_df.append(dotdict(
                id=id, particle_type=p.name,
                P=metric[0], T=metric[1], hit=metric[2], miss=metric[3], fp=metric[4],
            )) 
    print('')

    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby('particle_type').agg('sum').drop(columns=['id'])
    gb.loc[:, 'precision'] = gb['hit'] / gb['P']
    gb.loc[:, 'precision'] = gb['precision'].fillna(0)
    gb.loc[:, 'recall'] = gb['hit'] / gb['T']
    gb.loc[:, 'recall'] = gb['recall'].fillna(0)
    gb.loc[:, 'f-beta4'] = 17 * gb['precision'] * gb['recall'] / (16 * gb['precision'] + gb['recall'])
    gb.loc[:, 'f-beta4'] = gb['f-beta4'].fillna(0)

    gb = gb.sort_values('particle_type').reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, 'weight'] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb['f-beta4'] * gb['weight']).sum() / gb['weight'].sum()
    return gb, lb_score


def do_cv(cfg):
    path_to_model = sorted(glob(f'./working/train/{cfg.model_folder}/*.pth'))
    root = copick.from_file('./working/copick.config')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=n_classes,
        channels=(48, 64, 80, 80),
        strides=(2, 2, 1),
        num_res_units=1,
    ).to(device)

    models = []
    for p in path_to_model:
        model.load_state_dict(torch.load(p))
        model.eval
        models.append(model)
    
    inference_transforms = Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS")
    ])

    sigma = cfg.patch_size//2-0
    weight = get_gaussian_weight(cfg.patch_size, sigma, 0).to('cuda')

    if len(models) == 1:
        runs = root.runs[2:3] #only test TS_6_4
    else:
        runs = root.runs

    with torch.no_grad():
        
        location_df = []
        inference_time = []
        for fold, run in enumerate(runs):

            print(f'** TEST {run.name} FOR CV **')
            start = time.time()

            tomo = run.get_voxel_spacing(10)
            tomo = tomo.get_tomogram('denoised').numpy()
            original_shape = tomo.shape
            tomo_patches, coordinates  = extract_3d_patches_minimal_overlap([tomo], cfg.patch_size, cfg.overlap)
            tomo_patched_data = [{"image": img} for img in tomo_patches]
            tomo_ds = CacheDataset(data=tomo_patched_data, transform=inference_transforms, cache_rate=1.0)

            reconstructed = torch.zeros(
                [n_classes, original_shape[0], original_shape[1], original_shape[2]]
            ).to('cuda')  # To track overlapping regions
            count = torch.zeros(
                [n_classes, original_shape[0], original_shape[1], original_shape[2]]
            ).to('cuda')
            for i in range(len(tomo_ds)):
                if cfg.tta:
                    # w/o rotate
                    input_tensor = tomo_ds[i]['image'].unsqueeze(0).to('cuda')
                    model_output_tmp = models[fold](input_tensor)
                    model_output_tmp = model_output_tmp.squeeze(0)
                    model_outputs_tta = [model_output_tmp]
                    # tta with rotate90(k=1~3)
                    for k in range(1, cfg.tta_k_rotate+1):
                        input_tensor   = tomo_ds[i]['image'].to("cuda")
                        rotate         = Rotate90(k=k, spatial_axes=(0, 2))
                        rotate_inverse = Rotate90(k=4-k, spatial_axes=(0, 2))
                        input_tensor = rotate(input_tensor)
                        input_tensor = input_tensor.unsqueeze(0)
                        model_output_tmp = models[fold](input_tensor)
                        model_output_tmp = model_output_tmp.squeeze(0)
                        model_output_tmp = rotate_inverse(model_output_tmp)
                        model_outputs_tta.append(model_output_tmp)
                    model_output = torch.stack(model_outputs_tta, 0).mean(0)
                    model_output = model_output.unsqueeze(0)
                else:
                    input_tensor = tomo_ds[i]['image'].unsqueeze(0).to('cuda')
                    model_output = models[fold](input_tensor)

                prob = torch.softmax(model_output[0], dim=0) #prob.shape: (7,96,96,96)

                reconstructed[
                    :, 
                    coordinates[i][0]:coordinates[i][0] + cfg.patch_size,
                    coordinates[i][1]:coordinates[i][1] + cfg.patch_size,
                    coordinates[i][2]:coordinates[i][2] + cfg.patch_size
                ] += prob

                count[
                    :, 
                    coordinates[i][0]:coordinates[i][0] + cfg.patch_size,
                    coordinates[i][1]:coordinates[i][1] + cfg.patch_size,
                    coordinates[i][2]:coordinates[i][2] + cfg.patch_size
                ] += weight

            reconstructed /= count
            
            if isinstance(cfg.certainty_threshold, float):
                thresh_prob = reconstructed > cfg.certainty_threshold
                _, max_classes = thresh_prob.max(dim=0)
                thresh_max_classes = max_classes

            else:
                max_probs, max_classes = torch.max(reconstructed, dim=0)
                thresh_prob = torch.zeros_like(reconstructed)
                thresh_max_classes = torch.zeros_like(reconstructed[0])
                for ch in range(n_classes):
                    max_channel_is_one = torch.where(max_classes==ch, 1, 0)
                    thresh_prob[ch] = max_probs * max_channel_is_one > cfg.certainty_threshold[ch]
                    thresh_prob[ch] = torch.where(thresh_prob[ch]==1, ch, 0)
                    thresh_max_classes += thresh_prob[ch]

            thresh_max_classes = thresh_max_classes.cpu().numpy()
            
            location = {}
            for c in classes:
                cc = cc3d.connected_components(thresh_max_classes == c)
                stats = cc3d.statistics(cc)
                zyx=stats['centroids'][1:]*10.012444 #https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895#3040071
                zyx_large = zyx[stats['voxel_counts'][1:] > cfg.blob_threshold]
                xyz =np.ascontiguousarray(zyx_large[:,::-1])
                location[id_to_name[c]] = xyz

            df = dict_to_df(location, run.name)
            location_df.append(df)

            inference_time.append(time.time()-start)
        
        location_df = pd.concat(location_df)

    location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))
    # location_df.to_csv("submission.csv", index=False)

    #-- scoring
    gb, lb_score = compute_lb(location_df, f'{cfg.local_kaggle_dataset_dir}/train/overlay/ExperimentRuns')
    gb.to_csv(f'./working/train/{cfg.model_folder}/cv_lb={lb_score:.4f}_{cfg.certainty_threshold}_{cfg.blob_threshold}_tta={cfg.tta}.csv')
    print(gb)
    
    print()
    print('lb_score:',lb_score)

    mean_inference_time = sum(inference_time)/len(inference_time)
    print()
    print('inferences for cv done. ')
    print(
        f'mean inference time is {mean_inference_time:.2f} sec'
        # f'\nscoring time would be {mean_inference_time*500/60/60:.2f} hr'
    )    



if __name__ == '__main__':

    cfg = dotdict(load_config('config.yml'))

    if 1:
        seed_everything(cfg.seed)
        
        dt = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder_name_for_all_fold = f'{dt}_3dunet_{cfg.batch_size}_{cfg.lr}_{cfg.epochs}_{cfg.patch_size}x{cfg.patch_size}x{cfg.patch_size}'
        os.makedirs(f'./working/train/{output_folder_name_for_all_fold}', exist_ok=True)
        
        # cfg.data_split = generate_split_dataset()
        cfg.data_split = generate_split_dataset_submit()
        cfg.model_folder = output_folder_name_for_all_fold
        # pprint(cfg)

        print(f'\n** {dt} start training from here!! **')

        for i in [0,1,2,3,4,5,6]: 
        # for i in [2]:
            print('\n' + '-'*20 + f' fold{i} ' + '-'*20)
            run_train(cfg, fold=i, stage=1)
            run_train(cfg, fold=i, stage=2)
        
    # -- eval
    # cfg.based_on_fold2 = True
    # cfg.certainty_threshold = find_the_best_certainty_threshold(cfg)
    # do_cv(cfg)

    # print(f'\ncertainty thresh is {cfg.certainty_threshold}')