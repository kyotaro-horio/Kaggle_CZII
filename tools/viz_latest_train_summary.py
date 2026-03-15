import os
import matplotlib
matplotlib.use('TkAgg')  # Or 'QtAgg', depending on your environment

import matplotlib.pyplot as plt
import pandas as pd

from czii_helper.dataset import *
from czii_helper.helper import *

def get_latest_folder(parent_directory):
    entries = [os.path.join(parent_directory, entry) for entry in os.listdir(parent_directory)]
    directories = [entry for entry in entries if os.path.isdir(entry)]
    
    if not directories:
        raise AssertionError("No directories found.")
    
    latest_folder = max(directories, key=os.path.getmtime)
    return latest_folder


if __name__ == '__main__':

    config = dotdict(load_config('config.yml'))

    vi = config.val_interval
    latest_metrics_folder = get_latest_folder(f'{config.output_dir}/train')

    df = pd.read_csv(f'{latest_metrics_folder}/train_log.csv')


    fig, ax1 = plt.subplots(2, figsize=(8,6))

    # -- graph1. train/val loss and lb score
    m4_loss_train = np.poly1d(np.polyfit(df.epoch, df.loss_train, 4))
    m4_loss_val = np.poly1d(np.polyfit(df.epoch, df.loss_val, 4))
    polyline = np.linspace(0 + 1, df.epoch.to_list()[-1] + 1, 100)

    # ax1[0].set_title('epoch vs loss&metric')
    ax2 = ax1[0].twinx()

    ax1[0].plot(df.epoch, df.loss_train, color='blue', alpha=.2)
    ax1[0].plot(df.epoch, df.loss_val, color='red', alpha=.2)
    ax1[0].plot(polyline, m4_loss_train(polyline), '--', color='blue', label='train loss fit')
    ax1[0].plot(polyline, m4_loss_val(polyline), '--', color='red', label='val loss fit')

    ax2.plot(df.epoch, df.lb_score, color='green', label='val lb score')

    # ax1[0].set_xlabel('epoch')
    ax1[0].set_ylabel('loss')
    ax1[0].legend(loc='upper left')
    # ax1[0].grid()
    ax2.set_ylabel('val lb score')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='upper right')
    ax2.grid()

    # -- graph2. lr
    # ax1[1].set_title(f'lr scheduler (start from {config.lr})')
    # ax1[1].plot(df.epoch, df.lr)
    # ax1[1].set_xlabel('epoch')
    # ax1[1].set_ylabel('lr')

    # -- graph3. fbeta
    m4_a_fer = np.poly1d(np.polyfit(df.epoch, df['fbeta4_a-fer'], 4))
    m4_b_amy = np.poly1d(np.polyfit(df.epoch, df['fbeta4_b-amy'], 4))
    m4_b_gal = np.poly1d(np.polyfit(df.epoch, df['fbeta4_b-gal'], 4))
    m4_ribo  = np.poly1d(np.polyfit(df.epoch, df['fbeta4_ribo'], 4))
    m4_thyr  = np.poly1d(np.polyfit(df.epoch, df['fbeta4_thyr'], 4))
    m4_vlp   = np.poly1d(np.polyfit(df.epoch, df['fbeta4_vlp'], 4))
    polyline = np.linspace(0 + 1, df.epoch.to_list()[-1] + 1, 100)

    # ax1[1].set_title('fbeta4 for each class')
    ax1[1].plot(df.epoch, df['fbeta4_a-fer'], alpha=.2)
    ax1[1].plot(df.epoch, df['fbeta4_b-amy'], alpha=.2)
    ax1[1].plot(df.epoch, df['fbeta4_b-gal'], alpha=.2)
    ax1[1].plot(df.epoch, df['fbeta4_ribo'], alpha=.2)
    ax1[1].plot(df.epoch, df['fbeta4_thyr'], alpha=.2)
    ax1[1].plot(df.epoch, df['fbeta4_vlp'], alpha=.2)
    ax1[1].plot(polyline, m4_a_fer(polyline), '--', label='a-fer(1)')
    ax1[1].plot(polyline, m4_b_amy(polyline), '--', label='b-amy(0)')
    ax1[1].plot(polyline, m4_b_gal(polyline), '--', label='b-gal(2)')
    ax1[1].plot(polyline, m4_ribo(polyline), '--', label='ribo(1)')
    ax1[1].plot(polyline, m4_thyr(polyline), '--', label='thyr(2)')
    ax1[1].plot(polyline, m4_vlp(polyline), '--', label='vlp(1)')

    ax1[1].set_xlabel('epoch')
    ax1[1].set_ylabel('fbeta4')
    ax1[1].set_ylim([0, 1])
    ax1[1].legend(loc='upper left')
    ax1[1].grid()
    # --

    fig.tight_layout()

    plt.savefig(f'{latest_metrics_folder}/summary.png') 
    plt.show()



