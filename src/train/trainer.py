import os
import torch
import numpy as np
import csv
from monai.data import decollate_batch

from train.metric import calc_fbeta


def train(
        cfg, train_loader, val_loader, 
        model, loss_func, metric_func, optimizer, scheduler, 
        post_pred, post_label, 
        ):
    
    # writing header in the command line and log file
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

    # setting up max epochs and val interval
    max_epochs = 100 if cfg.stage == 2 else cfg.epochs
    val_interval = cfg.val_interval

    # main train loop
    best_metric, best_metric_epoch = -1, -1
    for epoch in range(max_epochs):
        model.train()

        epoch_loss, step = 0, 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(cfg.device)
            labels = batch_data["label"].to(cfg.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            epoch_loss_val, step_val = 0, 0
            with torch.no_grad():
                for val_data in val_loader:
                    step_val += 1
                    val_inputs = val_data["image"].to(cfg.device)
                    val_labels = val_data["label"].to(cfg.device)
                    val_outputs = model(val_inputs)

                    val_loss = loss_func(val_outputs, val_labels)
                    epoch_loss_val += val_loss.item()

                    metric_val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    metric_val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    # computing metric at the current iteration
                    metric_func(y_pred=metric_val_outputs, y=metric_val_labels)

                epoch_loss_val /= step_val
                metrics = metric_func.aggregate(reduction="mean_batch")
                metric = torch.mean(metrics).numpy(force=True)
                # metric = (metrics[0]*1 + metrics[1]*0 + metrics[2]*2 + metrics[3]*1 + metrics[4]*2 + metrics[5]*1) / 7 # weighted dice metric for czii
                metric_func.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), f"./working/train/{cfg.model_folder}/{cfg.exp_name}_{cfg.fold}.pth")

                # writing training infos in the command linea and log file               
                print(
                    f"{epoch + 1:0>3}/{max_epochs:0>3} "
                    f"| {current_lr:.7f} "
                    f"| {epoch_loss:.4f} "
                    f"| {epoch_loss_val:.4f} "
                    f"| {metrics[0]:.4f} {metrics[1]:.4f} {metrics[2]:.4f} {metrics[3]:.4f} {metrics[4]:.4f} {metrics[5]:.4f} "
                    f"| {metric:.4f} "
                    f"| {best_metric:.4f} ({best_metric_epoch:0>3} epoch)"
                    )
                metrics = [float(m) for m in metrics]
                metric = float(metric)
                with open(f"./working/train/{cfg.model_folder}/log_stage{cfg.stage}_fold{cfg.fold}.csv", 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                            epoch + 1, current_lr, epoch_loss, epoch_loss_val, 
                            metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], 
                            metric, 
                        ])

