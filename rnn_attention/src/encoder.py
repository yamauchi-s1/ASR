import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    '''
    Encoder
    
    dim_in : 入力特徴量の次元数
    dim_hidden : 隠れ層の次元数
    dim_proj : Projection層の次元数
    num_layers : RNN層の数
    sub_sample :レイヤーごとに設定するフレームの動き
                num_layer=4の時、subsamle=[1,2,3,1]とすると
                2層目でフレーム数を1/2にし、３層目で1/3にする。合計1/6
                
    rnn_type : LSTM or GRU
    '''
    
    def __init__(self,
                dim_in,
                dim_hidden,
                dim_proj,
                num_layers=2,
                bidirectional=True, 
                sub_sample=None, 
                rnn_type='LSTM'):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        rnn = []
        
        for n in range(self.num_layers):
            input_size = dim_in if n == 0 else dim_proj     
            if rnn_type == 'GRU':
                rnn.append(nn.GRU(input_size=input_size,
                                  hidden_size=dim_hidden,
                                  bidirectional=bidirectional,
                                  batch_first=True))
            else:
                rnn.append(nn.LSTM(input_size=input_size, 
                                   hidden_size=dim_hidden,
                                   num_layers=1, 
                                   bidirectional=bidirectional, 
                                   batch_first=True))
                
        self.rnn = nn.ModuleList(rnn)
        if sub_sample is None:
            # 定義されていない場合はフレームの間引きを行わない
            self.sub_sample = [1 for _ in range(num_layers)]
        else:
            self.sub_sample = sub_sample
            
        proj = []
        for n in range(self.num_layers):
            input_size = dim_hidden * (2 if bidirectional else 1)
            proj.append(nn.Linear(input_size, dim_proj))
            
        self.proj = nn.ModuleList(proj)
        
    
    def forward(self, sequence, lengths):
        '''
        ネットワーク計算
        sequence : 各発話の入力系列 [ B x T x D]
        lengths : 各発話の系列長 [B]
            []はテンソルサイズ
            B: ミニバッチ内の発話数(ミニバッチ)
            T: 入力テンソルの系列長(ゼロパディングを含める)
            D: 入力次元(dim_in)
        '''
        
        output = sequence
        # lengthsをCPU上のint64に変換
        output_lengths = lengths.cpu().long()
        
        for n in range(self.num_layers):
            rnn_input = pack_padded_sequence(output, 
                                             output_lengths,
                                             batch_first=True)
            
            output, (h, c) = self.rnn[n](rnn_input)
            # RNN層からProjection層へ入力するためにtensorデータに戻す
            output, output_lengths = pad_packed_sequence(output, batch_first=True)
            
            sub = self.sub_sample[n]
            if sub > 1:
                # 間引きを実行
                output = output[:, ::sub] # 全ての行を選択し、subステップごとに列を選択
                output_lengths = torch.div((output_lengths + 1), sub, rounding_mode='floor')
                
            output = torch.tanh(self.proj[n](output))
            
        return output, output_lengths