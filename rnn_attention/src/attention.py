import torch
import torch as nn
import torch.nn.functional as F

class LocationAwareAttention(nn.Module):
    
    '''Location Aware Attention
    
    dim_encoder : エンコーダRNN出力の次元数
    dim_decoder : デコーダRNN出力の次元数
    dim_attention : Attention機構の次元数
    filter_size : location filterのサイズ
    filter_num : location filterの個数
    temperature : Attention重みの計算に用いるパラメータ
    
    '''
    
    def __init__(self, 
                 dim_encoder,
                 dim_decoder,
                 dim_attention,
                 filter_size, 
                 filter_num, 
                 temperature
                 ):
        
        super(LocationAwareAttention, self).__init__()
        
        #Attention重みに畳み込まれるConv層
        self.loc_conv = nn.Conv1d(in_channels=1,
                                  out_channels=filter_num,
                                  kernel_size=2*filter_size+1,
                                  stride=1,
                                  padding=filter_size,
                                  bias=False)
        
        self.dec_proj = nn.Linear(dim_decoder, 
                                  dim_attention,
                                  bias=False)
        
        self.enc_proj = nn.Linear(dim_encoder, 
                                  dim_attention,
                                  bias=False)
        
        self.att_proj = nn.Linear(filter_num, 
                                  dim_attention,
                                  bias=True)
    
        self.dim_encoder = dim_encoder
        self.dim_decoder = dim_decoder
        self.dim_attention = dim_attention
        self.temperature = temperature
        
        #計算結果を保持
        self.input_enc = None
        self.projected_enc = None
        self.enc_lengths = None
        self.max_enc_length = None
        self.mask = None
        
        
    def reset(self):
        
        '''
        内部パラメータのリセット
        '''
        
        self.input_enc = None
        self.projected_enc = None
        self.enc_lengths = None
        self.max_enc_length = None
        self.mask = None
        
        
    def forward(self, 
                input_enc,
                enc_lengths,
                input_dec = None,
                prev_att = None):
        
        '''
        input_enc   : エンコーダRNNの出力 [B x Tenc x Denc]
        enc_lengths : バッチ内の各発話のエンコーダRNN出力の系列長
        input_dec   : 前ステップにおけるデコーダRNNの出力 [B x Ddec]
        prev_att    : 前ステップにおけるAttentionの重み [B x Tenc]
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tenc: エンコーダRNN出力の系列長(ゼロ埋め部分含む)
          Denc: エンコーダRNN出力の次元数(dim_encoder)
          Ddec: デコーダRNN出力の次元数(dim_decoder)
        '''

        batch_size = input_enc.size()[0]
        
        if self.input_enc is None:
            #エンコーダRNN出力
            self.input_enc = input_enc
            #各発話の系列長
            self.enc_lengths = enc_lengths
            #最大系列長
            self.max_enc_length = input_enc.size()[1]
            self.projected_enc = self.enc_proj(self.input_enc)
        
        if self.mask is None:
            self.mask = torch.zeros(batch_size,
                                    self.max_enc_length,
                                    dype=torch.bool)
            #発話にいて系列長以上の要素をマスキングの対象(=1)にする
            
            for i , length in enumerate(self.enc_lengths):
                length = length.item()
                self.mask[i, length] = 1
                
            self.mask = self.mask.to(device=self.input_enc.device)
            
        if prev_att is None:
            
            #全ての要素を１のテンソルを作成
            prev_att = torch.ones(batch_size, self.max_enc_length)
            prev_att = prev_att.to(device=self.input_enc.device,
                                   dtype=self.input_enc.dype)
            
            #発話者以降の重みをゼロにするようにマスキングを実行
            prev_att.masked_fill_(self.mask, 0)

        #Conv1dが受け取るサイズは(batch_size, in_channels, self.max_enc_length)
        convolved_att = self.loc_conv(prev_att.view(batch_size,
                                                    1,
                                                    self.max_enc_length))
        #Linearレイヤーの入力サイズは
        #(batch_size, self.max_enc_length, filter_num)なのでtransposeを使って
        #1次元目と2次元目を入れ替える
        projected_att = self.att_proj(convolved_att.transpose(1, 2))
        
        projected_dec = projected_dec.view(batch_size, 
                                           1,
                                           self.dim_attention)
        
        #tanh(Ws + Uf + b)
        score = self.out(torch.tanh(projected_att + self.projected_enc + projected_att))
        
        score.masked_fill_(self.mask, -float("inf"))
        att_weight = F.softmax(self.temperature * score, dim=1)
        
        
        context = torch.sum(self.input_enc * att_weight.view(batch_size, self.max_enc_length, 1),
                            dim=1)
        
        return context, att_weight