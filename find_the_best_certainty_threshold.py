import os
import shutil
from datetime import datetime
import copick
from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from glob import glob
from pprint import pprint

from monai.networks.nets import UNet, DynUNet
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
)

import cc3d
import pandas as pd
from scipy.optimize import linear_sum_assignment

from train.trainer import *
from czii_helper.helper import *
from czii_helper.helper2 import *
from czii_helper.dataset import *

cls_names = ['', 'a-fer', 'b-amy', 'b-gal', 'ribo ', 'thyr ', 'vlp  ']

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


def compute_lb_for_exp(submit_df, overlay_dir):
    valid_id = list(submit_df['experiment'].unique())
    # print(valid_id)

    eval_df = []
    for id in valid_id:
        truth = read_one_truth(id, overlay_dir) #=f'{valid_dir}/overlay/ExperimentRuns')
        id_df = submit_df[submit_df['experiment'] == id]
        for p in PARTICLE:
            p = dotdict(p)
            # print('\r', id, p.name, end='', flush=True)
            xyz_truth = truth[p.name]
            xyz_predict = id_df[id_df['particle_type'] == p.name][['x', 'y', 'z']].values
            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius* 0.5)
            eval_df.append(dotdict(
                id=id, particle_type=p.name,
                P=metric[0], T=metric[1], hit=metric[2], miss=metric[3], fp=metric[4],
            ))
    # print('')
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

    return gb


def find_the_best_certainty_threshold(cfg):

    input_dir  = cfg.local_kaggle_dataset_dir
    path_to_model = sorted(glob(f'./working/train/{cfg.model_folder}/*.pth'))
    root = copick.from_file("./working/copick.config")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('./working',exist_ok=True)
    print(f'\n** experiment for thresh starts w/ model {cfg.model_folder}!! **\n')

    # Non-random transforms to be cached
    inference_transforms = Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS")
    ])

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(root.pickable_objects)+1,
        channels=(48, 64, 80, 80),
        strides=(2, 2, 1),
        num_res_units=1,
    ).to(device)
# ---

    models = []
    for p in path_to_model:
        print(f'load {os.path.basename(p)}')
        model.load_state_dict(torch.load(p))
        model.eval()
        models.append(model)
    
    if cfg.based_on_fold2:
        models = models[2:3]

    # -- set threshold list for this experiment
    step = 0.05 #0.05
    thresh_list = [int(step*i*100)/100 for i in range(1, int(1/step))] 
    print('\nthreshold steps:', thresh_list)

    import csv
    with open(f'./working/train/{cfg.model_folder}/experiment_for_finding_the_best_certainty_threshold.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['fold', 'cls_id'] + [str(th) for th in thresh_list]     
        writer.writerow(header)

        with torch.no_grad():
            
            max_fbeta_thresh_list = []
            for fold, run in enumerate(root.runs):

                if cfg.based_on_fold2 and run.name != 'TS_6_4': # for just testing fold2 #todo!!
                    continue

                tomo = run.get_voxel_spacing(10)
                tomo = tomo.get_tomogram('denoised').numpy()
                tomo_patches, coordinates  = extract_3d_patches_minimal_overlap([tomo], cfg.patch_size, cfg.overlap)
                tomo_patched_data = [{"image": img} for img in tomo_patches]
                tomo_ds = CacheDataset(data=tomo_patched_data, transform=inference_transforms, cache_rate=1.0)

                print('\n', '-'*20, f'fold{fold} {run.name} {os.path.basename(path_to_model[fold])}', '-'*20, '\n')
                print('cls | name  |         certainty thresh           | fbeta4')
                print('    |       | 0    1    2    3    4    5    6    |       ')
                print('====|=======|====================================|========')
                print('0   | BG    |------------------------------------| SKIPPED')
                #      1   | a-fer | 1.00 0.05 1.00 1.00 1.00 1.00 1.00 | 0.81

                max_fbeta_thresh = [0.5]
                # for target_cls in [1,2]:
                for target_cls in classes:
                    print('----|-------|------------------------------------|-------')
                    #      1   | a-fer | 1.00 0.05 1.00 1.00 1.00 1.00 1.00 | 0.8095
                    
                    fbeta_list  = [fold, target_cls, ]
                    best_thresh = 0.05
                    best_fbeta  = -1
                    for th in thresh_list:
                        
                        location_df = []
                        CERTAINTY_THRESHOLD = [1, ] #background is not considered in this exp
                        
                        for i in range(1, 6+1):
                            if i == target_cls:
                                CERTAINTY_THRESHOLD.append(th)
                            else:
                                CERTAINTY_THRESHOLD.append(1)
                        
                        pred_masks = []
                        for i in range(len(tomo_ds)):
                            input_tensor = tomo_ds[i]['image'].unsqueeze(0).to("cuda")
                            
                            model_output = model(input_tensor)
                            prob = torch.softmax(model_output[0], dim=0) #prob.shape: (7,96,96,96)
                            
                            max_probs, max_classes = torch.max(prob, dim=0)
                            thresh_prob = torch.zeros_like(prob)
                            thresh_max_classes = torch.zeros_like(prob[0])
        
                            for ch in range(n_classes):
                                max_channel_is_one = torch.where(max_classes==ch, 1, 0)
                                thresh_prob[ch] = max_probs * max_channel_is_one > CERTAINTY_THRESHOLD[ch]
                                thresh_prob[ch] = torch.where(thresh_prob[ch]==1, ch, 0)
                                thresh_max_classes += thresh_prob[ch]

                            pred_masks.append(thresh_max_classes.cpu().numpy())

                        reconstructed_mask = reconstruct_array(pred_masks, coordinates, tomo.shape)

                        location = {}
                        for c in classes:
                            cc = cc3d.connected_components(reconstructed_mask == c)
                            stats = cc3d.statistics(cc)
                            zyx=stats['centroids'][1:]*10.012444 #https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895#3040071
                            zyx_large = zyx[stats['voxel_counts'][1:] > cfg.blob_threshold]
                            xyz =np.ascontiguousarray(zyx_large[:,::-1])

                            location[id_to_name[c]] = xyz

                        df = dict_to_df(location, run.name)
                        location_df.append(df)
                        location_df = pd.concat(location_df)
                        location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))

                        gb = compute_lb_for_exp(location_df, f'{input_dir}/train/overlay/ExperimentRuns')
                        fbeta = gb.iloc[target_cls-1]['f-beta4']
                        
                        text = f'{target_cls}   | {cls_names[target_cls]} | {CERTAINTY_THRESHOLD[0]:.2f} {CERTAINTY_THRESHOLD[1]:.2f} {CERTAINTY_THRESHOLD[2]:.2f} {CERTAINTY_THRESHOLD[3]:.2f} {CERTAINTY_THRESHOLD[4]:.2f} {CERTAINTY_THRESHOLD[5]:.2f} {CERTAINTY_THRESHOLD[6]:.2f} | {fbeta:.4f}'

                        if fbeta > best_fbeta:
                            best_thresh = CERTAINTY_THRESHOLD[target_cls]
                            best_fbeta  = fbeta
                            text += ' BEST!!'

                        print(text)
                        fbeta_list.append(fbeta)

                    max_fbeta_thresh.append(best_thresh)
                    writer.writerow(fbeta_list)
            
                max_fbeta_thresh_list.append(max_fbeta_thresh)
    #             print(f'\n--\nfold{fold} done! ')
    #             print(f'the best thresh list is {max_fbeta_thresh}')
    
    # print('\n--\nall experiment has done!!')
    # print('best thresh for each model is')
    # pprint(max_fbeta_thresh_list)
    
    return max_fbeta_thresh_list


if __name__=='__main__':
    
    cfg = dotdict(load_config('config.yml'))
    cfg.based_on_fold2 = False
    cfg.overlap = [1,1,1]

    res = find_the_best_certainty_threshold(cfg)

    if cfg.based_on_fold2:
        print(f'\n--\nfold2 done! ')
        print(f'the best thresh list is {res[0]}')
    else:
        print('\n--\nall experiment has done!!')
        print('best thresh for each model is')
        pprint(res)