import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src import levenshtein, SequneceDataModule, E2EModelLightningModule
import os
import numpy as np
import json

# 基本設定
unit = 'phone'

# 実験ディレクトリ
exp_dir = './exp_train_small'
feat_dir_test = '../data/fbank/test'

# 学習済みモデルのファイルパス
model_dir = os.path.join(exp_dir, unit + '_model_attention')
model_file = os.path.join(model_dir, 'best-checkpoint.ckpt')
config_file = os.path.join(model_dir, 'config.json')
with open(config_file, 'r') as f:
    config = json.load(f)

# 評価データの特徴量リストファイル
feat_scp_test = os.path.join(feat_dir_test, 'feats.scp')
label_test = os.path.join(exp_dir, 'data', unit, 'label_test')

# 平均・標準偏差ファイルとトークンリストのパス
mean_std_file = os.path.join(model_dir, 'mean_std.txt')
token_list_path = os.path.join(exp_dir, 'data', unit, 'token_list')

# テスト結果出力ディレクトリ
output_dir = os.path.join(exp_dir, unit + '_model_attention')
os.makedirs(output_dir, exist_ok=True)

# データモジュールのインスタンスを作成
data_module = SequneceDataModule(
    feat_scp_train=config['feat_scp_train'],
    label_train=config['label_train'],
    feat_scp_dev=config['feat_scp_dev'],
    feat_scp_test=config['feat_scp_test'],
    label_dev=config['label_dev'],
    label_test=config['label_test'],
    mean_std_file=config['mean_std_file'],
    token_list_path=config['token_list_path'],
    batch_size=config['batch_size']
)

# configをE2EModelLightningModuleに渡す
model = E2EModelLightningModule.load_from_checkpoint(
    checkpoint_path=model_file,
    config=config,  # configを辞書形式で渡す
)

# 'token_list' を `hparams` に追加
model.hparams.token_list = data_module.token_list
model.hparams.eos_id= data_module.eos_id


# PyTorch Lightning Trainerの設定
trainer = pl.Trainer(
    accelerator='gpu', 
    devices=[0],
)

# テストの実行
trainer.test(model, datamodule=data_module)