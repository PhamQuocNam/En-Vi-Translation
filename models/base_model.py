import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from config import Config
config = Config()

class encoder(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings= vocab_size, embedding_dim=config.embedding_dim, padding_idx=config.padding_idx)
        self.positional_encoding = nn.Embedding(config.max_seq_length,embedding_dim=config.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model= config.embedding_dim, nhead=config.n_heads, dim_feedforward=4092, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=config.num_layers)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.embedding_dim,768),
            nn.LeakyReLU(),
            nn.Linear(768,1024)
        )
    
    def forward(self, inputs):
        size= inputs.shape[1]
        mask = torch.zeros((size,size), device=inputs.device).type(torch.bool)
        padding_mask= (inputs==1)
        
        embedded_inputs = self.embedding(inputs)
        positions = torch.arange(size, device = inputs.device).unsqueeze(0)
        embedding_position = self.positional_encoding(positions)
        
        embedded_inputs += embedding_position
        outputs= self.encoder(embedded_inputs,mask =mask, src_key_padding_mask =padding_mask) # NxLxC
        
        outputs= self.fc(outputs) # NxLxC
        
        return outputs
        
        
    

class decoder(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding =nn.Embedding(num_embeddings= vocab_size, embedding_dim=1024, padding_idx=config.padding_idx)
        self.positional_encoding = nn.Embedding(num_embeddings=config.max_seq_length, embedding_dim=1024 )
        decoder_layer = nn.TransformerDecoderLayer(d_model=1024,nhead=config.n_heads, dim_feedforward=2048, dropout=0.2, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.fc = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(1024,2048),
            nn.LeakyReLU(),
            nn.Linear(2048,vocab_size)
        )
    
    
    def forward(self, tgt_inputs, encoder_outputs):
        size= tgt_inputs.shape[1]
        mask = (torch.triu(torch.ones((size,size), device=encoder_outputs.device)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        tgt_padding_mask = (tgt_inputs == 1) 
        
        embedded_tgt_inputs = self.embedding(tgt_inputs)
        positions = torch.arange(config.max_seq_length -1, device= tgt_inputs.device).unsqueeze(0)
        embedding_position = self.positional_encoding(positions)
        embedded_tgt_inputs+= embedding_position
        
        outputs = self.decoder(embedded_tgt_inputs,encoder_outputs, tgt_mask =mask, tgt_key_padding_mask = tgt_padding_mask  )
        outputs= self.fc(outputs)
        
        return outputs
        
        
        
    
    
class BaseModel(nn.Module):
    def __init__(self, config, en_tokenizer, vi_tokenizer):
        super().__init__()
        self.bos_idx = en_tokenizer.convert_tokens_to_ids('[BOS]')
        self.encoder = encoder(en_tokenizer.vocab_size,config)
        self.decoder = decoder(vi_tokenizer.vocab_size,config)
        
    def forward(self, input_ids , labels):
        encoder_outputs = self.encoder(input_ids)
        logits= self.decoder(labels, encoder_outputs)

        return logits.permute(0,2,1)