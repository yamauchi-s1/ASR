import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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
                enc_num_layers,
                enc_bidirectional,
                enc_sub_sample,
                enc_rnn_type,
                ):
        super(CTCModel, self).__init__()
        
        self.encoder = Encoder(dim_in = dim_in,
                                dim_hidden = dim_enc_hid,
                                dim_proj = dim_enc_proj,
                                num_layers = enc_num_layers,
                                bidirectional = enc_bidirectional,
                                sub_sample = enc_sub_sample,
                                rnn_type = enc_rnn_type)
        #出力層
        self.out = nn.Linear(in_features=dim_enc_proj,
                            out_features=dim_out)
        
        #重みを初期化
        lecun_initialization(self)
        
        
    def forward(self, input_sequence, input_lengths):
        
        '''
        forward計算
        --------------------------------------------------
        input_sequence : 各発話の入力系列([ B x Tin x D])
        input_lengths  : 各発話の系列長(フレーム数 [B])
        
        B   : ミニバッチ内の発話数
        Tin : 入力テンソルの系列長(0 paddingを含める)
        D   : 入力次元数(dim_in)
        --------------------------------------------------
        '''
        
        enc_out, enc_lenghts = self.encoder(input_sequence, input_lengths)
        output = self.out(enc_out)
        
        return output, enc_lenghts
        
