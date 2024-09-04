import torch
import torch.nn as nn
import torch.nn.functional as F

class LocationAwareAttention(nn.Module):
    '''Location Aware Attention'''

    def __init__(self, 
                 dim_encoder,
                 dim_decoder,
                 dim_attention,
                 filter_size, 
                 filter_num, 
                 temperature):
        super(LocationAwareAttention, self).__init__()
        
        # Attention重みに畳み込まれるConv層
        self.loc_conv = nn.Conv1d(in_channels=1,
                                  out_channels=filter_num,
                                  kernel_size=2*filter_size+1,
                                  stride=1,
                                  padding=filter_size,
                                  bias=False)
        
        self.dec_proj = nn.Linear(dim_decoder, dim_attention, bias=False)
        self.enc_proj = nn.Linear(dim_encoder, dim_attention, bias=False)
        self.att_proj = nn.Linear(filter_num, dim_attention, bias=True)
        self.out = nn.Linear(dim_attention, 1, bias=True)
    
        self.dim_encoder = dim_encoder
        self.dim_decoder = dim_decoder
        self.dim_attention = dim_attention
        self.temperature = temperature
        
        # 計算結果を保持
        self.input_enc = None
        self.projected_enc = None
        self.enc_lengths = None
        self.max_enc_length = None
        self.mask = None
        
    def reset(self):
        '''内部パラメータのリセット'''
        self.input_enc = None
        self.projected_enc = None
        self.enc_lengths = None
        self.max_enc_length = None
        self.mask = None
        
    def forward(self, input_enc, enc_lengths, input_dec=None, prev_att=None):
        '''
        input_enc   : エンコーダRNNの出力 [B x Tenc x Denc]
        enc_lengths : バッチ内の各発話のエンコーダRNN出力の系列長
        input_dec   : 前ステップにおけるデコーダRNNの出力 [B x Ddec]
        prev_att    : 前ステップにおけるAttentionの重み [B x Tenc]
        '''

        batch_size = input_enc.size(0)
        
        if self.input_enc is None:
            # エンコーダRNN出力
            self.input_enc = input_enc
            # 各発話の系列長
            self.enc_lengths = enc_lengths
            # 最大系列長
            self.max_enc_length = input_enc.size(1)
            self.projected_enc = self.enc_proj(self.input_enc)
            
        if input_dec is None:
            input_dec = torch.zeros(batch_size, self.dim_decoder, dtype=input_enc.dtype, device=input_enc.device)
        
        # 前のデコーダRNN出力を射影する
        projected_dec = self.dec_proj(input_dec)
        
        if self.mask is None:
            self.mask = torch.zeros(batch_size, self.max_enc_length, dtype=torch.bool, device=input_enc.device)
            for i, length in enumerate(self.enc_lengths):
                length = length.item()
                self.mask[i, length:] = 1
            
        if prev_att is None:
            prev_att = torch.ones(batch_size, self.max_enc_length, dtype=input_enc.dtype, device=input_enc.device)
            prev_att.masked_fill_(self.mask, 0)

        # Conv1dが受け取るサイズは (batch_size, in_channels, self.max_enc_length)
        convolved_att = self.loc_conv(prev_att.view(batch_size, 1, self.max_enc_length))
        
        # Linearレイヤーの入力サイズは (batch_size, self.max_enc_length, filter_num)
        projected_att = self.att_proj(convolved_att.transpose(1, 2))
        
        projected_dec = projected_dec.view(batch_size, 1, self.dim_attention)
        
        # tanh(Ws + Uf + b)
        score = self.out(torch.tanh(projected_dec + self.projected_enc + projected_att))
        score = score.view(batch_size, self.max_enc_length)
        score.masked_fill_(self.mask, -float("inf"))
        att_weight = F.softmax(self.temperature * score, dim=1)
        
        context = torch.sum(self.input_enc * att_weight.view(batch_size, self.max_enc_length, 1), dim=1)
        
        return context, att_weight