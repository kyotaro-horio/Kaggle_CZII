import os
import shutil

from datetime import datetime

import copick

from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write
from collections import defaultdict

from tqdm import tqdm
import numpy as np

from monai.networks.nets import UNet
from monai.networks.nets import DynUNet

from monai.data import DataLoader, Dataset, CacheDataset
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

import cc3d
import pandas as pd
from scipy.optimize import linear_sum_assignment

from train.trainer import *
from czii_helper.helper import *
from czii_helper.helper2 import *

from czii_helper.dataset import *


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


if __name__ == '__main__':
    
    config = dotdict(
        load_config('config.yml')
    )

    input_dir  = config.local_kaggle_dataset_dir
    output_dir = config.output_dir
    os.makedirs(output_dir,exist_ok=True)

    path_to_model = f'./working/train/{config.model_folder}/best_metric_model.pth'

    # -- get copick root
    copick_config_path = f"{output_dir}/copick.config"
    root = copick.from_file(copick_config_path)

    # -- get tomograms and their segmentation mask arrays
    data_dicts = []
    for run in tqdm(root.runs):
        tomogram = run.get_voxel_spacing(config.voxel_size).get_tomogram(config.tomo_type).numpy()
        segmentation = run.get_segmentations(name=config.copick_segmentation_name, user_id=config.copick_user_name, voxel_size=config.voxel_size, is_multilabel=True)[0].numpy()
        data_dicts.append({"image": tomogram, "label": segmentation})
    
    #set up dataloaders
    train_files, val_files, test_files = data_dicts[:2] + data_dicts[4:], data_dicts[3: 4], data_dicts[2: 3]

    # Non-random transforms to be cached
    inference_transforms = Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS")
    ])
    
# --
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print()
    print(
        f'dataset: '
        f'\n\ttrain {len(train_files)} samples' 
        f'\n\tval   {len(val_files)} samples'
        f'\n\ttest  {len(test_files)} samples')
    print(
        f'device:'  
        f'\n\t{device}')
    print()

    if config.model == 'unet':
        # Create UNet, DiceLoss and Adam optimizer
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=len(root.pickable_objects)+1,
            channels=(48, 64, 80, 80),
            strides=(2, 2, 1),
            num_res_units=1,
        ).to(device)

    elif config.model == 'dynunet':
        # model = DynUNet(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=len(root.pickable_objects)+1,
        #     kernel_size=[3, 3, 3, 3, 3],
        #     strides=[1, 2, 2, 2, 2],
        #     upsample_kernel_size=[2, 2, 2, 2],
        #     filters=[32, 64, 128, 256, 320],
        #     # deep_supr_num=3,  # Add deep supervision
        # ).to(device)
        
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
            kernel_size=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2],
            filters=[32, 64, 96, 128],
            # deep_supr_num=3,  # Add deep supervision
            # res_block=True, 
        ).to(device)
    
    else:
        raise AssertionError('MODEL NOT FOUND!!')
# -- 

    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    dt = datetime.now().strftime('%Y%m%d_%H%M%S')
    print()
    print('-'*30, f'{dt} start test for 1fold from here!!', '-'*30)
    print()

    with torch.no_grad():
        location_df = []
        for run in root.runs:
            if run.name != 'TS_6_4': continue

            tomo = run.get_voxel_spacing(10)
            tomo = tomo.get_tomogram(config.tomo_type).numpy()

            tomo_patches, coordinates  = extract_3d_patches_minimal_overlap([tomo], config.patch_size)

            tomo_patched_data = [{"image": img} for img in tomo_patches]

            tomo_ds = CacheDataset(data=tomo_patched_data, transform=inference_transforms, cache_rate=1.0)

            pred_masks = []

            for i in range(len(tomo_ds)):

                if config.tta:
                    # tta with rotate90(k=1~3)
                    model_outputs_tta = []
                    for k in range(1, 3+1):
                        input_tensor   = tomo_ds[i]['image']
                        rotate         = Rotate90(k=k, spatial_axes=(0, 2))
                        rotate_inverse = Rotate90(k=4-k, spatial_axes=(0, 2))
                        
                        input_tensor = rotate(input_tensor)
                        input_tensor = input_tensor.unsqueeze(0).to("cuda")
                        
                        tmp_model_output = model(input_tensor)
                        tmp_model_output = tmp_model_output.squeeze(0)
                        
                        tmp_model_output = rotate_inverse(tmp_model_output)
                        
                        model_outputs_tta.append(tmp_model_output)
                    
                    model_output = torch.stack(model_outputs_tta, 0).mean(0)
                    model_output = model_output.unsqueeze(0)
                
                else:
                    input_tensor   = tomo_ds[i]['image'].unsqueeze(0).to('cuda')
                    model_output = model(input_tensor)

                prob = torch.softmax(model_output[0], dim=0) #prob.shape: (7,96,96,96)
                
                if isinstance(config.certainty_threshold, float):
                    thresh_prob = prob > config.certainty_threshold
                    _, max_classes = thresh_prob.max(dim=0)

                    pred_masks.append(max_classes.cpu().numpy())

                else:
                    max_probs, max_classes = torch.max(prob, dim=0)
                    
                    thresh_prob = torch.zeros_like(prob)
                    thresh_max_classes = torch.zeros_like(prob[0])
                    for ch in range(n_classes):
                        max_channel_is_one = torch.where(max_classes==ch, 1, 0)
                        thresh_prob[ch] = max_probs * max_channel_is_one > config.certainty_threshold[ch]
                        thresh_prob[ch] = torch.where(thresh_prob[ch]==1, ch, 0)
                        thresh_max_classes += thresh_prob[ch]

                    pred_masks.append(thresh_max_classes.cpu().numpy())


            reconstructed_mask = reconstruct_array(pred_masks, coordinates, tomo.shape)
            
            location = {}

            for c in classes:
                cc = cc3d.connected_components(reconstructed_mask == c)
                stats = cc3d.statistics(cc)
                zyx=stats['centroids'][1:]*10.012444 #https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895#3040071
                zyx_large = zyx[stats['voxel_counts'][1:] > config.blob_threshold]
                xyz =np.ascontiguousarray(zyx_large[:,::-1])

                location[id_to_name[c]] = xyz

            df = dict_to_df(location, run.name)
            location_df.append(df)
        
        location_df = pd.concat(location_df)

    location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))

    # -- scoring 

    gb, lb_score = compute_lb(location_df, f'{input_dir}/train/overlay/ExperimentRuns')
    gb.to_csv(f'./working/train/{config.model_folder}/1fold_lb={lb_score:.4f}_{config.certainty_threshold}_{config.blob_threshold}_tta={config.tta}.csv')
    print(gb)
    
    print()
    print('lb_score:',lb_score)
    