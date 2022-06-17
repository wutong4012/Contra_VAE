import torch
import torch.nn as nn


class ContraVAEModel(nn.Module):
    """VAE with normal prior"""
    def __init__(self, encoder, decoder, config):
        super(ContraVAEModel, self).__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        
        if config.model_params.freeze_decoder:
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False
            
        self.latent_dim = config.model_params.latent_dim
        self.hidden_dim = config.model_params.hidden_dim
        self.embed_dim = config.model_params.embed_dim

        # Standard Normal prior
        loc = torch.zeros(self.latent_dim)
        scale = torch.ones(self.latent_dim)
        self.prior = torch.distributions.normal.Normal(loc, scale)
        
        if self.config.model_params.add_mlp:
            self.emb2hid = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
            self.feature_extractor = self.create_feature_extractor() 
            self.hid2lat = nn.Linear(self.hidden_dim, self.latent_dim, bias=False)
        else:
            self.emb2lat = nn.Linear(self.embed_dim, self.latent_dim, bias=False)
        
        self.lat2emb = nn.Linear(self.latent_dim, self.embed_dim, bias=False)
        
    def create_feature_extractor(self):
        return nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.ReLU(),
            ])
    
    def projection(self, hidden_fea):
        z = self.emb2hid(hidden_fea)
        for i in range(self.config.model_params.mlp_num):
            z = self.feature_extractor(z)
        return z

    def connect(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, latent_dim]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, latent_dim)
        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

        # (batch, nsamples, latent_dim)
        z = self.reparameterize(mean, logvar, nsamples)
        # Calculate the KL divergence of the normal distribution (with independent components)
        # and the standard normal distribution
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def connect_deterministic(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, latent_dim]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, latent_dim)
        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        logvar.fill_(.0)
        
        # (batch, nsamples, latent_dim)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        config:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, latent_dim)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, latent_dim)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, latent_dim)
        """
        batch_size, latent_dim = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, latent_dim)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, latent_dim)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def forward(self, encode_inputs, decode_inputs,
                pad_token_id=0, fb_mode=0, feats_embeds=None):
        attention_mask = (encode_inputs > 0).int()
        pooled_hidden_fea = self.encoder(encode_inputs, attention_mask)[1] # [bs, hz]
        
        if self.config.model_params.add_mlp:
            pooled_hidden_fea = self.projection(pooled_hidden_fea) # [bs, hz]
            feats = self.hid2lat(pooled_hidden_fea) # [bs, lz]
        else:
            feats = self.emb2lat(pooled_hidden_fea) # [bs, lz]
            
        pooled_hidden_fea = self.lat2emb(feats) # [bs, hz]
        feats_embeds = pooled_hidden_fea.unsqueeze(1).repeat(1, encode_inputs.size(1), 1) # [bs, seq_len, hz]
        # mask the padding is 0
        feats_embeds = feats_embeds * attention_mask.unsqueeze(2).repeat(1, 1, pooled_hidden_fea.size(1))
        
        if fb_mode in [0,1]:
            latent_z, loss_kl = self.connect(pooled_hidden_fea)
            latent_z = latent_z.squeeze(1)

            # Decoding
            outputs = self.decoder(input_ids=decode_inputs, past=latent_z, 
                                   labels=decode_inputs, label_ignore=pad_token_id,
                                   feats_embeds=feats_embeds)
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

        elif fb_mode==2: 
            # no variance
            latent_z, loss_kl = self.connect_deterministic(pooled_hidden_fea)
            latent_z = latent_z.squeeze(1)
            
            # Decoding
            outputs = self.decoder(input_ids=decode_inputs, past=latent_z, 
                                   labels=decode_inputs, label_ignore=pad_token_id,
                                   feats_embeds=feats_embeds)
            loss_rec = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
        
        if self.config.model_params.length_weighted_loss:
            reconstrution_mask=(decode_inputs != pad_token_id).int() # the padding token for GPT2
            sent_length = torch.sum(reconstrution_mask, dim=1)
            loss_rec /= sent_length
            
        return loss_rec.mean(), loss_kl.mean(), feats
