import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

# https://zenn.dev/yuto_mo/articles/6e2803495029d4
if __name__=='__main__':

    cfg = dotdict(load_config('config.yml'))
    init_lr = float(cfg.lr)

    model = nn.Linear(10,1)
    optimizer = optim.Adam(model.parameters(), init_lr)  
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(cfg.epochs*0.3), round(cfg.epochs*0.8)], gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs // 3)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / cfg.epochs) ** 0.9)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=init_lr*0.01, max_lr=init_lr, step_size_up=200, mode='exp_range', gamma=0.98)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=init_lr*0.01, last_epoch=-1) #epoch 700

    lrs = []
    for epoch in range(cfg.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        scheduler.step()

    plt.figure(figsize=(10,4))
    plt.plot(lrs)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.grid(True)
    plt.show()
