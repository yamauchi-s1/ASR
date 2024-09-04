import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch 
import torch.nn as nn
from attention import LocationAwareAttention
import numpy as np
import matplotlib.pyplot as plt

#Attentin Rnnのデコーダ

class Decoder(nn.Module):
    ''' デコーダ
    dim_in:          入力系列(=エンコーダ出力)の次元数
    dim_hidden:      デコーダRNNの次元数
    dim_out:         出力の次元数(sosとeosを含む全トークン数)
    dim_att:         Attention機構の次元数
    att_filter_size: LocationAwareAttentionのフィルタサイズ
    att_filter_num:  LocationAwareAttentionのフィルタ数
    sos_id:          <sos>トークンの番号
    att_temperature: Attentionの温度パラメータ
    num_layers:      デコーダRNNの層の数
    '''
    def __init__(self,
                 dim_in,
                 dim_hidden,
                 dim_out,
                 dim_att,
                 att_filter_size, 
                 att_filter_num,
                 sos_id,
                 att_temperature=1.0,
                 num_layers=1):

        super(Decoder, self).__init__()

        # <sos>と番号を設定
        self.sos_id = sos_id

        # 入力次元数と出力次元数
        self.dim_in = dim_in
        self.dim_out = dim_out

        # 1ステップ前の出力を入力するembedding層
        # (次元数がdim_outのベクトルから次元数が
        #  dim_hiddenのベクトルに変換する)
        self.embedding = nn.Embedding(dim_out, dim_hidden)
        
        # Location aware attention
        self.attention = LocationAwareAttention(dim_in,
                                                dim_hidden, 
                                                dim_att,
                                                att_filter_size, 
                                                att_filter_num,
                                                att_temperature)

        # RNN層
        # RNNには1ステップ前の出力(Embedding後)と
        # エンコーダ出力(Attention後)が入力される．
        # よってRNNの入力次元数は
        # dim_hidden(Embedding出力の次元数) \
        #   + dim_in(エンコーダ出力の次元数)
        self.rnn = nn.LSTM(input_size=dim_hidden+dim_in, 
                           hidden_size=dim_hidden,
                           num_layers=num_layers, 
                           bidirectional=False,
                           batch_first=True)

        # 出力層
        self.out = nn.Linear(in_features=dim_hidden,
                             out_features=dim_out)

        # Attention重み行列(表示用)
        self.att_matrix = None


    def forward(self, enc_sequence, enc_lengths, label_sequence=None):
        batch_size = enc_sequence.size()[0]
        
        if label_sequence is not None:
            # 学習時:
            #   = ラベル情報が与えられている場合は
            #     ラベル系列長を使う
            max_step = label_sequence.size()[1]
        else:
            # 評価時:
            #   = ラベル情報が与えられていない場合は
            #     エンコーダ出力系列長を使う
            max_step = enc_sequence.size()[1]
        
        # エンコーダ出力のデバイスを取得
        device = enc_sequence.device

        # 1ステップ前のトークン．初期値は <sos> とする
        prev_token = torch.ones(batch_size, 1, dtype=torch.long, device=device) * self.sos_id
        
        # 1ステップ前のRNN出力とAttention重みをNoneで初期化する
        prev_rnnout = None
        prev_att = None
        # 1ステップ前のRNN内部パラメータ(h, c)もNoneで初期化する
        prev_h_c = None
        # Attentionの内部パラメータをリセットする
        self.attention.reset()

        # 出力テンソルを用意 [batch_size x max_step x dim_out]
        output = torch.zeros(batch_size, max_step, self.dim_out, device=device)

        # 表示用Attention重み行列の初期化
        self.att_matrix = torch.zeros(batch_size, max_step, enc_sequence.size(1), device=device)

        # 最大ステップの数だけデコーダを動かす
        for i in range(max_step):
            # Attentionを計算し，コンテキストベクトルと，Attention重みを得る
            context, att_weight = self.attention(enc_sequence, enc_lengths, prev_rnnout, prev_att)
            
            # RNNを1ステップ分動かす
            prev_token_emb = self.embedding(prev_token)

            # RNNに入力するためのベクトル結合
            context = context.view(batch_size, 1, self.dim_in)
            rnn_input = torch.cat((prev_token_emb, context), dim=2)
            
            # RNNに通す
            rnnout, h_c = self.rnn(rnn_input, prev_h_c)

            # RNN出力を線形層に通す
            out = self.out(rnnout)

            # 出力テンソルにoutを格納
            output[:, i, :] = out.view(batch_size, self.dim_out)
            
            # 1ステップ前のRNN出力とRNN内部パラメータ，Attention重み，トークンを更新する
            prev_rnnout = rnnout
            prev_h_c = h_c
            prev_att = att_weight

            # トークンの更新
            if label_sequence is not None:
                prev_token = label_sequence[:, i].view(batch_size, 1).to(device)
            else:
                _, prev_token = torch.max(out, 2)

            # 表示用Attention重み行列
            self.att_matrix[:, i, :] = att_weight

        return output

    def save_att_matrix(self, utt, filename):
        ''' Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        att_mat = self.att_matrix[utt].detach().numpy()
        # プロットの描画領域を作成
        plt.figure(figsize=(5,5))
        # カラーマップのレンジを調整
        att_mat -= np.max(att_mat)
        vmax = np.abs(np.min(att_mat)) * 0.0
        vmin = - np.abs(np.min(att_mat)) * 0.7
        # プロット
        plt.imshow(att_mat, 
                   cmap = 'gray',
                   vmax = vmax,
                   vmin = vmin,
                   aspect = 'auto')
        # 横軸と縦軸のラベルを定義
        plt.xlabel('Encoder index')
        plt.ylabel('Decoder index')

        # プロットを保存する
        plt.savefig(filename)
        plt.close()
        
            
            
        
        
        