import pandas as pd
import numpy as np
import torch
from config import Config
from utils import get_data, get_logger, data_cleaning, preprocess, data_handling, create_dict, get_tokenizers, \
    dataset, get_dataloader, training_visualizing, get_lr, en_punctuation_handling
from models.base_model import BaseModel 
import os
from tqdm import tqdm
from config import Config
logger = get_logger()
config= Config()
class Predictor:
    def __init__(self):
        self.model = BaseModel()
        try:
            self.model.load_state_dict(torch.load(config.pretrained_weight))
            logger.info(f"Loaded model weights from {config.pretrained_weight}")
        except Exception as e:
            logger.error(f"Failed to load model weights: {str(e)}")
            raise TypeError("Failed to load model weights")
        
        self.model.load_state_dict(torch.load(config.pretrained_weight))
        self.model.eval()
        self.en_tokenizer, self.vi_tokenizer = get_tokenizers(config.checkpoint_dir)
        
    
    def predict(self, sentence):
        input = en_punctuation_handling(sentence)
        input_ids = self.en_tokenizer(input,truncation=True,max_length=config.max_seq_length,padding='max_length',return_tensors='pt' )['input_ids']
        input_ids = input_ids.to(config.device)
        
        translated_sentence= '[BOS]'
        with torch.no_grad():
            for i in range(config.max_seq_length):
                target = self.vi_tokenizer(translated_sentence, truncation=True, max_length = config.max_seq_length, padding='max_length', return_tensors='pt')['input_ids']
                target = target.to(config.device)
                preds = self.model(input_ids,target)
                token_idx = preds.argmax(1)[:,i]
                word = self.vi_tokenizer.convert_ids_to_tokens(token_idx)[-1]
                translated_sentence += ' '+ word
                
                if word =='[EOS]':
                    break
                
        final_translation = translated_sentence.replace('[BOS]', '').replace('[EOS]', '').strip()
        logger.info('Done!')
        print(f'Origin: {sentence}')
        print(f'Translation: {final_translation}')
        
        
    
