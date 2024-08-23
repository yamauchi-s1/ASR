import torch.nn as nn
from Encoder import Encoder
from initialize import lecun_initialization
import numpy as np

class CTCModel(nn.Module):
    
    '''
    CTCモデルの定義
    -----------------------------------------------------------------------
    dim_in            : 入力次元数
    dim_enc_hid       : エンコーダーの隠れ層
    dim_enc_proj      : エンコーダーの出力次元数(projection層の次元数)
    dim_out           : 出力の次元数
    enc_num_layers    : エンコーダのレイヤー数
    enc_bidirectional : Trueでエンコーダーにbidirectional RNNを使用
    enc_sub_sample    : エンコーダにおいてレイヤーごとに設定するフレームの間引き率
    enc_rnn_type      : エンコーダRNNの種類 ['LSTM', 'GRU']を設定する
    ----------------------------------------------------------------------
    '''
    
    def __init__(self, 
                dim_in, 
                dim_enc_hid,
                dim_enc_proj,
                dim_out,
                emc_num_layers,
                enc_bidirectional,
                enc_sub_sample,
                enc_rnn_type,
                ):
        super(CTCModel, self).__init__()
        
        self.encoder = Encoder()
        