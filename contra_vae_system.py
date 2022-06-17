import torch
import numpy as np

from brownian_bridge_loss import BrownianBridgeLoss
from contra_vae_model import ContraVAEModel
from contra_vae_dataset import ContraVAEDataset
from contra_vae_dataloader import create_dataloader
from pytorch_lightning.core.lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_transformers import (BertTokenizer,
                                  BertConfig, BertForLatentConnector,
                                  GPT2Config, GPT2ForLatentConnector)


class ContraVAESystem(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_size = config.model_params.latent_dim
        self.train_steps = config.exp_params.train_steps
        self.batch_size = config.data_params.batch_size
        self.learning_rate = config.optim_params.pretrain_lr
        self.beta_t_list = self.frange_cycle_zero_linear(self.train_steps, start=0.0, 
                                                         stop=config.model_params.loss_beta_m,  
                                                         n_cycle=10)
        self.train_bb_loss = []
        
        self._set_model()
        self._set_dataset()
        
    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config, 
                                 self.pad_token_id, self.batch_size)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config,
                                 self.pad_token_id, self.batch_size, shuffle=False)

    def _set_dataset(self):
        self.train_dataset = ContraVAEDataset(
            train=True,
            config=self.config
        )
        self.test_dataset = ContraVAEDataset(
            train=False,
            config=self.config
        )
        
    def _set_model(self):
        self.encoder_tokenizer = BertTokenizer.from_pretrained(self.config.model_params.encoder_model_path)
        encoder_config = BertConfig.from_pretrained(self.config.model_params.encoder_model_path)
        encoder = BertForLatentConnector.from_pretrained(self.config.model_params.encoder_model_path,
                                                         config=encoder_config, 
                                                         latent_size=self.latent_size)

        # The decoder pretraining use the bert tokenizer.
        decoder_tokenizer = BertTokenizer.from_pretrained(self.config.model_params.decoder_model_path)
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>'}
        decoder_tokenizer.add_special_tokens(special_tokens_dict)
        decoder_config = GPT2Config.from_pretrained(self.config.model_params.decoder_model_path)
        decoder = GPT2ForLatentConnector.from_pretrained(self.config.model_params.decoder_model_path,
                                                         config=decoder_config, 
                                                         latent_size=self.latent_size)
        
        self.pad_token_id = decoder_tokenizer.pad_token_id
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        decoder.resize_token_embeddings(len(decoder_tokenizer))

        self.model = ContraVAEModel(encoder, decoder, self.config)
        if self.config.model_params.load_model_path is not None:
            load_model = torch.load(self.config.model_params.load_model_path)
            new_dict = {key[len('model.'):]:val for key, val in load_model['state_dict'].items()}
            self.model.load_state_dict(new_dict)
            print("Load Model from {} !".format(self.config.model_params.load_model_path))
        
        # for k,v in self.model.named_parameters():
        #     print(k)
        # total_num = sum(p.numel() for p in self.model.parameters())
        # trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('Total:', total_num, 'Trainable:', trainable_num)
    
    def mask_tokens(self, inputs):
        ''' Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. '''
        # We sample a few tokens in each sequence for masked-LM training 
        # (with probability config.model_params.mlm_prob defaults to 0.15 in Bert/RoBERTa)

        masked_indices = torch.bernoulli(
            torch.full(inputs.shape, self.config.model_params.mlm_prob)).to(torch.uint8)

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).to(torch.uint8) & masked_indices
        inputs[indices_replaced] = self.encoder_tokenizer.convert_tokens_to_ids(
            self.encoder_tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(inputs.shape, 0.5)).to(torch.uint8) & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.encoder_tokenizer), inputs.shape, dtype=torch.long).to(inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs

    def frange_cycle_zero_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=4):
        L = np.ones(n_iter) * stop
        period = n_iter/n_cycle
        step = (stop-start)/(period*self.config.optim_params.ratio_increase) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                if i < period*self.config.optim_params.ratio_zero:
                    L[int(i+c*period)] = start
                else: 
                    L[int(i+c*period)] = v
                    v += step
                i += 1
        return L 
    
    def cal_0tT_loss(self, encode_inputs, decode_inputs, beta_t, attr:str):
        if not self.config.model_params.only_bb_loss:
            attention_mask = (encode_inputs > 0).int()
            encode_inputs = self.mask_tokens(encode_inputs) * attention_mask # mask the padding
        recon_loss, kl_loss, feats = self.model(encode_inputs, decode_inputs, self.pad_token_id)
        loss = recon_loss + beta_t * kl_loss
        self.log(attr + '_loss', loss)
        self.log(attr + '_recon_loss', recon_loss)
        self.log(attr + '_kl_loss', beta_t * kl_loss)
        
        return loss, feats
    
    def get_batch_losses(self, batch, mode:str):
        torch.cuda.empty_cache()
        beta_t = self.beta_t_list[self.global_step]
        z0_loss, z0_feats = self.cal_0tT_loss(batch['z0_encode'], batch['z0_decode'], beta_t, attr='z0_'+mode)
        zt_loss, zt_feats = self.cal_0tT_loss(batch['zt_encode'], batch['zt_decode'], beta_t, attr='zt_'+mode)
        zT_loss, zT_feats = self.cal_0tT_loss(batch['zT_encode'], batch['zT_decode'], beta_t, attr='zT_'+mode)
        
        loss_fn = BrownianBridgeLoss(
            z_0=z0_feats, z_t=zt_feats, z_T=zT_feats,
            t_=batch['t_'], t=batch['t'], T=batch['T'],
            alpha=0, var=0,
            eps=self.config.model_params.eps,
            max_seq_len=batch['total_t'].float(),
        )
        
        bb_loss = loss_fn.get_loss()
        loss = (z0_loss + zt_loss + zT_loss) / 3 + bb_loss # TODO
        
        # print('z_loss', (z0_loss + zt_loss + zT_loss) / 3)
        # print('bb_loss', bb_loss)
        
        self.log(mode + '_beta_t', beta_t)
        self.log(mode + '_bb_loss', bb_loss.item())
        self.log(mode + '_all_loss', loss.item())
        
        if self.config.model_params.only_bb_loss:
            return bb_loss
        else:
            return loss

    def training_step(self, batch, batch_idx):
        # for idx, pg in enumerate(self.optimizers().param_groups):
        #     print(f"learning_rate_{idx}", pg['lr'])
        return self.get_batch_losses(batch, 'train')

    def test_step(self, batch, batch_idx):
        return self.get_batch_losses(batch, 'test')

    def configure_optimizers(self):
        if self.config.optim_params.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.learning_rate)
            print('Using the adamw optimizer!')
        elif self.config.optim_params.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.learning_rate)
            print('Using the adam optimizer!')
        elif self.config.optim_params.optimizer == 'sgd':
            # Note the difference between 'model.parameters()' and 'model.named_parameters()'
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.learning_rate,
                                        momentum=0.9)
            print('Using the sgd optimizer!')
        
        if self.config.optim_params.scheduler == 'warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=self.config.optim_params.warmup_steps,
                                                        num_training_steps=self.train_steps)
            print('Using the warmup scheduler!')
        elif self.config.optim_params.scheduler == 'plateau':
            # Note that need to add 'mode'
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.8)
            print('Using the plateau scheduler!')

        # Must be written strictly according to the specification! ! !
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
                'monitor': self.config.optim_params.monitor
            }
        }
            
