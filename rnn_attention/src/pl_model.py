import torch
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
from model import MyE2EModel
import levenshtein

class E2EModelLightningModule(pl.LightningModule):
    
    def __init__(self, config):
        super(E2EModelLightningModule, self).__init__()
        
        self.save_hyperparameters(config)
        
        self.model = MyE2EModel(
            dim_in=config['dim_in'],
            dim_enc_hid=config['enc_hidden_dim'],
            dim_enc_proj=config['enc_projection_dim'], 
            dim_dec_hid=config['dec_hidden_dim'],
            dim_out=config['num_tokens'],
            dim_att=config['att_hidden_dim'], 
            att_filter_size=config['att_filter_size'],
            att_filter_num=config['att_filter_num'],
            sos_id=config['sos_id'], 
            att_temperature=config['att_temperature'],
            enc_num_layers=config['enc_num_layers'],
            dec_num_layers=config['dec_num_layers'],
            enc_bidirectional=config['enc_bidirectional'],
            enc_sub_sample=config['enc_sub_sample'], 
            enc_rnn_type=config['enc_rnn_type']
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    
    def forward(self, input_sequence, input_lengths, label_sequnece=None):
        
        dec_out, enc_lengths = self.model(input_sequence, input_lengths, label_sequnece)
        return dec_out, enc_lengths
    
    def training_step(self, batch, batch_idx):
        # トレーニングステップの定義
        loss = self._shared_step(batch, batch_idx, phase='train')
        self.log('train_loss', loss, on_epoch=True)  # トレーニング損失のログ
        return loss

    def validation_step(self, batch, batch_idx):
        # バリデーションステップの定義
        loss = self._shared_step(batch, batch_idx, phase='validation')
        self.log('val_loss', loss, on_epoch=True)  # バリデーション損失のログ
        return loss

    def _shared_step(self, batch, batch_idx, phase):
        # バッチデータの取得
        features, labels, feat_lens, label_lens, utt_ids = batch

        # テンソルのデバイスを確認し、すべて同じデバイスに移動する
        device = features.device  # PyTorch Lightningが自動で設定するデバイスを取得

        # 必要に応じてデバイスを移動
        labels = labels.to(device)
        feat_lens = feat_lens.to(device)
        label_lens = label_lens.to(device)

        sorted_lens, indices = torch.sort(feat_lens.view(-1), descending=True)
        features = features[indices]
        labels = labels[indices]
        feat_lens = sorted_lens
        label_lens = label_lens[indices]

        # ラベルの末尾に<eos>を付与
        labels = torch.cat((labels, torch.zeros(labels.size(0), 1, dtype=torch.long, device=device)), dim=1)
        for m, length in enumerate(label_lens):
            labels[m][length] = self.hparams.sos_id
        label_lens += 1
        labels = labels[:, :torch.max(label_lens)]

        # モデルの出力を計算（フォワードパス）
        outputs, _ = self(features, feat_lens, labels)

        # 損失の計算
        b_size, t_size, _ = outputs.size()
        loss = self.criterion(outputs.view(b_size * t_size, -1), labels.reshape(-1))
        loss *= torch.mean(label_lens.float()).item()

        return loss
        
    
    def _calculate_errors(self, outputs, labels, label_lens, utt_ids, phase):
        # 認識エラーを計算する
        total_error = 0
        total_token_length = 0

        for n in range(outputs.size(0)):
            _, hyp_per_step = torch.max(outputs[n], 1)
            hyp_per_step = hyp_per_step.cpu().numpy()

            hypothesis = [self.hparams.token_list[m] for m in hyp_per_step[:label_lens[n]]]
            reference = [self.hparams.token_list[m] for m in labels[n][:label_lens[n]].cpu().numpy()]

            error, _, _, _, ref_length = levenshtein.calculate_error(hypothesis, reference)

            total_error += error
            total_token_length += ref_length

        # Attention重み行列を保存
        if phase == 'validation':
            self.model.save_att_matrix(0, f"{self.hparams.out_att_dir}/{utt_ids[0]}_ep{self.current_epoch+1}.png")

        return total_error, total_token_length

    def configure_optimizers(self):
        # オプティマイザーの設定
        optimizer = optim.Adadelta(self.parameters(), 
                                   lr=self.hparams['initial_learning_rate'], 
                                   rho=0.95, 
                                   eps=1e-8, 
                                   weight_decay=0.0)
        return optimizer