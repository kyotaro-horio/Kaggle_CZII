import os
import torch

from monai.data import decollate_batch

import numpy as np
import csv

from train.metric import calc_fbeta_metric_for_czii


def train(
        cfg, trn_loader, val_loader, 
        model, loss_function, metrics_function, optimizer, scheduler, 
        device, post_pred, post_label, 
        ):
    
    text = ''
    text +=   '                    | loss -----------| metric ----------------------------------------------------------------'
    text += '\nepoch   | lr        | train  | val    | a-fer  b-amy  b-gal  ribo   thyr   vlp    | mean   | best              '
    text += '\n========|===========|========|========|===========================================|========|==================='
    #          005/500 | 0.0000999 | 0.9091 | 0.9203 | 0.0062 0.0000 0.0000 0.0000 0.0545 0.0052 | 0.0172 | 0.0172 (005 epoch)
    print(text)

    with open(f"./working/train/{cfg.model_folder}/log_stage{cfg.stage}_fold{cfg.fold}.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
                'epoch', 'lr', 'loss_train', 'loss_val',
                'dice_a_fer', 'dice_b_amy', 'dice_b_gal', 'dice_ribo', 'dice_thyr', 'dice_vlp', 
                'dice_mean', 
            ])

    max_epochs = cfg.epochs
    if cfg.stage == 2:
        max_epochs = 100
    val_interval = cfg.val_interval
    best_metric = -1
    best_metric_epoch = -1
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in trn_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            epoch_loss_val = 0
            step_val = 0
            with torch.no_grad():
                for val_data in val_loader:
                    step_val += 1
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = model(val_inputs)

                    val_loss = loss_function(val_outputs, val_labels)
                    epoch_loss_val += val_loss.item()

                    metric_val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    metric_val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    # compute metric for current iteration
                    metrics_function(y_pred=metric_val_outputs, y=metric_val_labels)

                epoch_loss_val /= step_val
                metrics = metrics_function.aggregate(reduction="mean_batch")
                metric = torch.mean(metrics).numpy(force=True)
                # metric = (metrics[0]*1 + metrics[1]*0 + metrics[2]*2 + metrics[3]*1 + metrics[4]*2 + metrics[5]*1) / 7 # weighted dice metric for czii
                metrics_function.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), f"./working/train/{cfg.model_folder}/{cfg.experiment_name}_{cfg.fold}.pth")
                                    
                print(f"{epoch + 1:0>3}/{max_epochs:0>3} | {current_lr:.7f} | {epoch_loss:.4f} | {epoch_loss_val:.4f} | {metrics[0]:.4f} {metrics[1]:.4f} {metrics[2]:.4f} {metrics[3]:.4f} {metrics[4]:.4f} {metrics[5]:.4f} | {metric:.4f} | {best_metric:.4f} ({best_metric_epoch:0>3} epoch)")
                
                metrics = [float(m) for m in metrics]
                metric = float(metric)
                with open(f"./working/train/{cfg.model_folder}/log_stage{cfg.stage}_fold{cfg.fold}.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                            epoch + 1, current_lr, epoch_loss, epoch_loss_val, 
                            metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], 
                            metric, 
                        ])

