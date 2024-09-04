import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from src import SequenceDataset
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src import levenshtein, SequneceDataModule, E2EModelLightningModule
import json
import os
import sys
import shutil


# 基本設定
unit = 'phone'

# ディレクトリ設定
feat_dir_train = '../data/fbank/train_small'
feat_dir_dev = '../data/fbank/dev'
train_set_name = os.path.basename(feat_dir_train)
exp_dir = './exp_' + train_set_name

# ファイル設定
feat_scp_train = os.path.join(feat_dir_train, 'feats.scp')
feat_scp_dev = os.path.join(feat_dir_dev, 'feats.scp')
label_train = os.path.join(exp_dir, 'data', unit, 'label_' + train_set_name)
label_dev = os.path.join(exp_dir, 'data', unit, 'label_dev')
mean_std_file = os.path.join(feat_dir_train, 'mean_std.txt')
token_list_path = os.path.join(exp_dir, 'data', unit, 'token_list')
output_dir = os.path.join(exp_dir, unit + '_model_attention')
out_att_dir = os.path.join(output_dir, 'att_matrix')

# ディレクトリ作成
os.makedirs(out_att_dir, exist_ok=True)

# # 特徴量の平均/標準偏差ファイルを読み込む
# with open(mean_std_file, mode='r') as f:
#     lines = f.readlines()
#     mean_line = lines[1]
#     feat_mean = np.array(mean_line.split(), dtype=np.float32)
    

# データモジュールのインスタンスを作成
data_module = SequneceDataModule(
    feat_scp_train=feat_scp_train,
    label_train=label_train,
    feat_scp_dev=feat_scp_dev,
    label_dev=label_dev,
    mean_std_file=mean_std_file,
    token_list_path=token_list_path,
    batch_size=10
)

token_list = data_module.token_list
num_tokens = data_module.num_tokens
sos_id = data_module.sos_id
eos_id = data_module.eos_id
# 特徴量の次元数を定義
feat_dim = np.size(data_module.feat_mean)


# 設定を辞書形式にする
config = {
    # データ設定
    'unit'           : unit,
    'feat_scp_train' : feat_scp_train,
    'feat_scp_dev'   : feat_scp_dev,
    'label_train'    : label_train,
    'label_dev'      : label_dev,
    'mean_std_file'  : mean_std_file,
    'token_list_path': token_list_path,
    'output_dir'     : output_dir,
    'out_att_dir'    : out_att_dir,
    'dim_in'         : feat_dim,  # 特徴量の次元数
    'num_tokens'     : num_tokens,  # トークン数
    'sos_id'         : sos_id,  # <sos>トークンのID
    
    # モデル設定
    'batch_size': 10,
    'max_num_epoch': 60,
    'enc_num_layers': 5,
    'enc_sub_sample': [1, 2, 2, 1, 1],
    'enc_rnn_type': 'GRU',
    'enc_hidden_dim': 320,
    'enc_projection_dim': 320,
    'enc_bidirectional': True,
    'dec_num_layers': 1,
    'dec_hidden_dim': 300,
    'att_hidden_dim': 320,
    'att_filter_size': 100,
    'att_filter_num': 10,
    'att_temperature': 2.0,

    # トレーニング設定
    'initial_learning_rate': 1.0,
    'clip_grad_threshold': 5.0,
    'lr_decay_start_epoch': 7,
    'lr_decay_factor': 0.5,
    'early_stop_threshold': 3,
    'evaluate_error': {'train': False, 'validation': True}
}

# 設定をJSON形式で保存する
conf_file = os.path.join(output_dir, 'config.json')
with open(conf_file, mode='w') as f:
    json.dump(config, f, indent=4)

print(f"Config saved to {conf_file}")

# ニューラルネットワークモデルを作成する
model = E2EModelLightningModule(
    config
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',  # バリデーション損失が監視対象
    patience=3,  # 損失が改善しない場合、3エポックで停止
    verbose=True
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # バリデーション損失が監視対象
    dirpath=config['output_dir'],  # モデルを保存するディレクトリ
    filename='best-checkpoint',  # 保存ファイル名
    save_top_k=1,  # 最も良いモデルのみ保存
    mode='min'  # 最小化する場合の基準（損失を最小化）
)

trainer = pl.Trainer(
    max_epochs=config['max_num_epoch'],
    devices=1 if torch.cuda.is_available() else None,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    precision=16 if torch.cuda.is_available() else 32,
    logger=True,
    callbacks=[early_stop_callback, checkpoint_callback]
)

# 学習の実行
trainer.fit(model, data_module)