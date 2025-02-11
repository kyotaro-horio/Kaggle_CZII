import os
import shutil
from tqdm import tqdm
import numpy as np

from collections import defaultdict

import copick
from copick_utils.segmentation import segmentation_from_picks
import copick_utils.writers.write as write

from czii_helper.helper import *


if __name__ == '__main__':
    
    cfg = dotdict(load_config('config.yml'))
    input_dir           = cfg.local_kaggle_dataset_dir
    output_dir          = './working'
    copick_config_path  = f"{output_dir}/copick.config"
    output_overlay      = f"{output_dir}/overlay"
    source_dir          = f'{input_dir}/train/overlay'
    destination_dir     = f'{output_dir}/overlay'
    os.makedirs(output_dir, exist_ok=True)

    radius_factor = cfg.radius_factor # 0.8
    copick_user_name = 'copickUtils'
    copick_segmentation_name = 'paintedPicks'
    voxel_size = 10
    tomo_type = 'denoised'
    
    # Walk through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding subdirectories in the destination
        relative_path = os.path.relpath(root, source_dir)
        target_dir = os.path.join(destination_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy and rename each file
        for file in files:
            if file.startswith("curation_0_"):
                new_filename = file
            else:
                new_filename = f"curation_0_{file}"
            
            # Define full paths for the source and destination files
            source_file = os.path.join(root, file)
            destination_file = os.path.join(target_dir, new_filename)
            
            # Copy the file with the new name
            shutil.copy2(source_file, destination_file)
            # print(f"Copied {source_file} to {destination_file}")


    root = copick.from_file(copick_config_path)

    # -- generate multi-calss segmentation masks from picks, and save them to the copick overlay directory once
    target_objects = defaultdict(dict)
    for object in root.pickable_objects:
        if object.is_particle:
            target_objects[object.name]['label'] = object.label
            target_objects[object.name]['radius'] = object.radius


    for run in tqdm(root.runs):
        tomo = run.get_voxel_spacing(10)
        tomo = tomo.get_tomogram(tomo_type).numpy()
        target = np.zeros(tomo.shape, dtype=np.uint8)
        for pickable_object in root.pickable_objects:
            pick = run.get_picks(object_name=pickable_object.name, user_id="curation")
            if len(pick):  
                target = segmentation_from_picks.from_picks(pick[0], 
                                                            target, 
                                                            target_objects[pickable_object.name]['radius'] * radius_factor,
                                                            target_objects[pickable_object.name]['label']
                                                            )
        write.segmentation(run, target, copick_user_name, name=copick_segmentation_name) 
