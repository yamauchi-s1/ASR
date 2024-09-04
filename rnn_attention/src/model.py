import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from initialize import lecun_initialization
import numpy as np


class MyE2EModel(nn.Module):
    ''' Attention RNN によるEnd-to-Endモデルの定義
    ---------------------------------------------------------------
    dim_in:            入力次元数
    dim_enc_hid:       エンコーダの隠れ層の次元数
    dim_enc_proj:      エンコーダのProjection層の次元数
                       (これがエンコーダの出力次元数になる)
    dim_dec_hid:       デコーダのRNNの次元数
    dim_out:           出力の次元数(sosとeosを含む全トークン数)
    dim_att:           Attention機構の次元数
    att_filter_size:   LocationAwareAttentionのフィルタサイズ
    att_filter_num:    LocationAwareAttentionのフィルタ数
    sos_id:            <sos>トークンの番号
    enc_bidirectional: Trueにすると，エンコーダに
                       bidirectional RNNを用いる
    enc_sub_sample:    エンコーダにおいてレイヤーごとに設定する，
                       フレームの間引き率
    enc_rnn_type:      エンコーダRNNの種類．'LSTM'か'GRU'を選択する
    ---------------------------------------------------------------
    '''

    def __init__(self, dim_in, dim_enc_hid, dim_enc_proj, 
                 dim_dec_hid, dim_out, dim_att, 
                 att_filter_size, att_filter_num,
                 sos_id, att_temperature=1.0,
                 enc_num_layers=2, dec_num_layers=2, 
                 enc_bidirectional=True, enc_sub_sample=None, 
                 enc_rnn_type='LSTM'):
        super(MyE2EModel, self).__init__()

        # エンコーダを作成
        self.encoder = Encoder(dim_in=dim_in, 
                               dim_hidden=dim_enc_hid, 
                               dim_proj=dim_enc_proj, 
                               num_layers=enc_num_layers, 
                               bidirectional=enc_bidirectional, 
                               sub_sample=enc_sub_sample, 
                               rnn_type=enc_rnn_type)
        
        # デコーダを作成
        self.decoder = Decoder(dim_in=dim_enc_proj, 
                               dim_hidden=dim_dec_hid, 
                               dim_out=dim_out, 
                               dim_att=dim_att, 
                               att_filter_size=att_filter_size, 
                               att_filter_num=att_filter_num, 
                               sos_id=sos_id, 
                               att_temperature=att_temperature,
                               num_layers=dec_num_layers)

        # LeCunのパラメータ初期化を実行
        lecun_initialization(self)


    def forward(self,
                input_sequence,
                input_lengths,
                label_sequence=None):

        enc_out, enc_lengths = self.encoder(input_sequence,
                                            input_lengths)

        # デコーダに入力する
        dec_out = self.decoder(enc_out,
                               enc_lengths,
                               label_sequence)

        # デコーダ出力とエンコーダ出力系列長を出力する
        return dec_out, enc_lengths


    def save_att_matrix(self, utt, filename):
        ''' Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        # decoderのsave_att_matrixを実行
        self.decoder.save_att_matrix(utt, filename)