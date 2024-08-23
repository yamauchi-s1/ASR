from torch.utils.data import Dataset
import numpy as np
import sys


class SequenceDataset(Dataset):
    '''
    feat_scp : 特徴量リストファイル
    label_scp : ラベルファイル
    feat_mean : 特徴量の平均ベクトル
    feat_std : 特徴量の標準偏差を並べたベクトル
    pad_index : バッチ化の際にフレーム数を合わせる
    splice : 前後のフレームを特徴量を結合する
    '''

    def __init__(self, 
                feat_scp,
                label_scp,
                feat_mean,
                feat_std,
                pad_index=0,
                splice=0):
        
        self.num_utts = 0
        self.id_list = []
        self.feat_list = []
        self.feat_len_list = []
        self.label_len_list = []

        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.feat_std[self.feat_std<1E-10] = 1E-10
        self.feat_dim = np.size(self.feat_mean)
        self.max_feat_len = 0
        self.max_label_len = 0
        self.pad_index = pad_index
        self.splice = splice #前後nフレームの特徴量を結合

        #特徴量リスt,ラベルを1行すつ読み込み情報を取得
        with open(feat_scp, mode='r') as file_f, open(label_scp, mode='r') as file_l:
            
            for (line_feats, line_label) in zip (file_f, file_l):

                parts_feats = line_feats.split()
                parts_label = line_label.split()

                #話者IDの特徴量とラベル番号が一致していなければエラー
                if parts_feats[0] != parts_label[0]:
                    sys.stderr.write('IDs of feat and label do not match!!!')
                    exit(1)

                self.id_list.append(parts_feats[0]) #発話IDを追加
                self.feat_list.append(parts_feats[1]) #特徴量のパスをリストに追加
                feat_len = np.int64(parts_feats[2])#フレーム数を追加
                self.feat_len_list.append(feat_len) 

                label = np.int64(parts_label[1:])
                self.label_list.append(label)
                self.label_len_list.append(len(label))

                self.num_utts += 1

        self.max_feat_len = np.max(self.feat_len_list)
        self.max_label_len = np.max(self.label_len_list)

        #ラベルデータの長さを最大フレームに合わせるためにpad_indexの値で埋める
        for n in range(self.num_utts):
            pad_len = self.max_label_len - self.label_len_list[n]
            #pad_indexの値で埋める
            self.label_list[n] = np.pad(self.label_list[n],
                                        [0, pad_len],
                                        mode='constant',
                                        constant_values=self.pad_index)
            

    def __len__(self):
        '''
        学習データのサンプル数を返す関数
        '''
        return self.num_utts
    
    def __getitem__(self, idx):
        """
        データを返す関数

        idx = 発話番号
        """
        feat_len = self.feat_len_list[idx]
        label_len = self.label_len_list[idx]
        
        feat = np.fromfile(self.feat_list[idx],
                            dtype=np.float32)
        feat = feat.reshape(-1, self.feat_dim)
        feat = (feat - self.feat_mean) / self.feat_std
        
        org_feat = feat.copy()
        for n in range(-self.splice, self.splice+1):
            tmp = np.roll(org_feat, n, axis=0)
            
            #前にずらした場合、終端nフレームを0にする
            if n < 0:
                tmp[n:0] = 0
            #後ろにずらした場合は始端nフレームを0にする 
            elif n > 0:
                tmp[:n] = 0
            else:
                continue
            
            feat = np.hstack([feat, tmp])
            
            
        pad_len = self.max_feat_len - feat_len
        feat = np.pad(feat, 
                        [(0, feat), (0, 0)],
                        mode='constant',
                        constant_values=0)
        
        label = self.label_list[idx]
        utt_id = self.id_list[idx]
        
        return (
            feat, 
            label,
            feat_len,
            label_len,
            utt_id
        )
