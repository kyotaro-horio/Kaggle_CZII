import warnings
warnings.filterwarnings("ignore")

import os
import shutil

from datetime import datetime

import copick

from tqdm import tqdm
import numpy as np
from glob import glob

from monai.networks.nets import UNet, DynUNet
from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import AsDiscrete

from torchinfo import summary

from train.trainer import *
from train.dataloader import *
from train.loss import FbetaLoss

from czii_helper.helper import *
from czii_helper.helper2 import *


def run_train(fold):
    
    config = dotdict(
        load_config('config.yml')
    )

    seed_everything(config.seed)
    
    input_dir           = config.local_kaggle_dataset_dir

    output_dir          = config.output_dir
    copick_config_path  = f"{output_dir}/copick.config"
    output_overlay      = f"{output_dir}/overlay"

    source_dir          = f'{input_dir}/train/overlay'
    destination_dir     = f'{output_dir}/overlay'

    os.makedirs(output_dir,exist_ok=True)
        
    root = copick.from_file(copick_config_path)

    # -- split all data into train, val, and test set
    all_data_path = sorted(glob(f'{output_dir}/overlay/ExperimentRuns/*'))
    n_fold = len(all_data_path)
    test_names = [os.path.basename(f) for f in all_data_path]
    val_names = test_names[1:] + test_names[:1]
    train_names = []
    for i in range(n_fold):
        tmp_train_names = test_names.copy()
        tmp_train_names.remove(test_names[i])
        tmp_train_names.remove(val_names[i])
        train_names.append(tmp_train_names)
    # -- 

    # -- get tomograms and their segmentation mask arrays
    train_data_dicts = []
    val_data_dicts = []
    test_data_dicts = []
    for id, run in enumerate(root.runs):
        tomogram = run.get_voxel_spacing(config.voxel_size).get_tomogram(config.tomo_type).numpy()
        segmentation = run.get_segmentations(
            name=config.copick_segmentation_name, 
            user_id=config.copick_user_name, 
            voxel_size=config.voxel_size, 
            is_multilabel=True
            )[0].numpy()
        
        if run.name in train_names[fold]:
            train_data_dicts.append({"image": tomogram, "label": segmentation})
        elif run.name in val_names[fold]:
            val_data_dicts.append({"image": tomogram, "label": segmentation})
        elif run.name in test_names[fold]:
            test_names.append({"image": tomogram, "label": segmentation})
        else:
            raise AssertionError(f'{run.name} is not in overlay')

    #set up dataloaders
    train_files, val_files, test_files = train_data_dicts, val_data_dicts, test_data_dicts
    train_loader, val_loader, train_ds, val_ds \
          = make_train_val_dataloaders(train_files, val_files, config)

# -- 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print()
    print(
        f'dataset (fold{fold}): '
        f'\n\ttrain {train_names[fold]}'
        f'\n\tval   {val_names[fold]}'
        f'\n\ttest  {test_names[fold]}'
        )
    print(
        f'model:'
        f'\n\t{config.model}'
        )
    print(
        f'device:'  
        f'\n\t{device}'
        )
    print()

    if config.model == 'unet':
        # Create UNet, DiceLoss and Adam optimizer
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=n_classes,
            channels=(48, 64, 80, 80),
            strides=(2, 2, 1),
            num_res_units=1,
        ).to(device)

    elif config.model == 'dynunet':
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

    lr = float(config.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr)  
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / config.epochs) ** 0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config.epochs*0.3, config.epochs*0.8], gamma=0.1)

    dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  # softmax=True for multiclass

# -- 

    dt = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('-'*30, f'{dt} start training from here!!', '-'*30)
    print()

    output_folder_name_for_all_fold = \
        f'{dt}_{config.model}' + \
            f'_{config.batch_size}_{lr}_{config.epochs}' + \
                f'_{config.patch_size}x{config.patch_size}x{config.patch_size}'
    
    output_dir_train = \
        f'{output_dir}/train/{output_folder_name_for_all_fold}/fold{fold}'
    
    os.makedirs(output_dir_train, exist_ok=True)

    # Log model summary.
    # with open(f"{output_dir_train}/model_summary.txt", "w") as f:
    #     f.write(str(summary(model)))

    post_pred = AsDiscrete(argmax=True, to_onehot=n_classes)
    post_label = AsDiscrete(to_onehot=n_classes)

    train(
        output_dir_train, config, 
        train_loader, val_loader, 
        model, loss_function, dice_metric, optimizer, scheduler, 
        device, 
        post_pred, post_label,
        )

if __name__ == '__main__':

    for i in range(7): run_train(i)