import nltk.translate.bleu_score
import pandas as pd
import numpy as np
import torch
from config import Config
from utils import get_data, get_logger, data_cleaning, preprocess, data_handling, create_dict, get_tokenizers, \
    MyDataset, get_dataloader, training_visualizing, get_lr, bleu_score
import nltk
from models.base_model import BaseModel 
import os
from tqdm import tqdm
logger = get_logger()
config = Config()

class Trainer():
    def __init__(self):
        data = get_data(config.data_path)
        train_df= data['train']
        val_df = data['valid']
        test_df = data['test']
        
        logger.info("Prepare for data cleaning!")
        train_df = data_cleaning(train_df)
        val_df = data_cleaning(val_df)
        
        logger.info("Prepare for data preprocessing!")
        train_df = data_handling(train_df)
        val_df =  data_handling(val_df)
        test_df = data_handling(test_df)
    
        if os.listdir(config.tokenizer_dict) ==  []:
            create_dict(train_df,config.vocab_size,min_frequency=3)
        
        self.en_tokenizer, self.vi_tokenizer = get_tokenizers()
        
        train_ds = preprocess(train_df, config.max_seq_length, self.en_tokenizer, self.vi_tokenizer)
        val_ds = preprocess(val_df, config.max_seq_length, self.en_tokenizer, self.vi_tokenizer)
        test_ds = preprocess(test_df, config.max_seq_length,self.en_tokenizer, self.vi_tokenizer)
        
        train_ds = MyDataset(train_ds)
        val_ds = MyDataset(val_ds)
        test_ds =MyDataset(test_ds)
        
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(train_ds, val_ds, test_ds, batch_size=64)
        
        self.model =BaseModel(config, self.en_tokenizer, self.vi_tokenizer)
        self.model.to(config.device)
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay= config.weight_decay, betas=(0.9, 0.98), eps=1e-9)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index= self.vi_tokenizer.convert_tokens_to_ids('[PADDING]'))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',factor=0.1, patience= 5)
        
        self.best_weights = None
        self.best_loss = float('inf')
        
    
    def run(self):
        try:
            checkpoint_path = f"checkpoints/{config.model_name}_best.pth"
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path))
                logger.info(f"Loaded existing model from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load existing model: {str(e)}")
            
        
        history = self.train()
        training_visualizing(history)
        torch.save(self.best_weights,f"checkpoints/{config.model_name}_{config.epochs}.pth")
        
        self.evaluate()
        
    
    def valid(self, current_lr):
        losses = []
   
        with torch.no_grad():
            self.model.eval()
            for idx, (src, tgt) in tqdm(enumerate(self.val_loader),desc="Validation"):
                src= src.to(config.device)
                tgt= tgt.to(config.device)
                
                decoder_inputs= tgt[:,:-1]
                labels = tgt[:,1:]
                logits = self.model(src,decoder_inputs)
                loss = self.criterion(logits.reshape(-1,logits.size(1)), labels.reshape(-1))
                losses.append(loss)

            avg_loss= sum(losses)/len(losses)
            self.scheduler.step(avg_loss)
            
            if avg_loss < self.best_loss:
                logger.info("Updating best weight!")
                self.best_loss = avg_loss
                self.best_weights = self.model.state_dict()
            
            if current_lr != get_lr(self.optimizer):
                logger.info("Learning rate decreased. Loading best model weights.")
                if self.best_weights is not None:
                    self.model.load_state_dict(self.best_weights)
            
            
        return avg_loss
    
    def train(self):
        total_train_loss=[]
        total_val_loss = []
        for epoch in tqdm(range(config.epochs), desc='Epoch'):
            training_losses= []
            self.model.train()
            current_lr = get_lr(self.optimizer)
            for idx, (src, tgt) in tqdm(enumerate(self.train_loader),desc="Training"):
                src= src.to(config.device)
                tgt= tgt.to(config.device)
                self.optimizer.zero_grad()
                
                
                decoder_inputs= tgt[:,:-1]
                labels = tgt[:,1:]
                logits = self.model(src,decoder_inputs)
                loss = self.criterion(logits.reshape(-1,logits.size(1)), labels.reshape(-1))
                
                training_losses.append(loss)
                loss.backward()
                self.optimizer.step()
            train_loss = sum(training_losses)/len(training_losses)
            valid_loss =self.valid(current_lr)
            total_train_loss.append(train_loss)
            total_val_loss.append(valid_loss)
            logger.info(f'EPOCH {epoch+1}/{config.epochs}')
            logger.info(f'Training Loss: {train_loss:.4f}')
            logger.info(f'Validation Loss: {valid_loss:.4f}')
            logger.info(f'Learning Rate: {get_lr(self.optimizer):.6f}')
        return {
            'training': total_train_loss,
            'valid': total_val_loss
        }
        
    
    def evaluate(self):
        """Evaluate the model on the test dataset"""
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
        
        self.model.eval()
        total_loss = 0
        batch_count = 0
        total_score = 0
        with torch.no_grad():
            for (src, tgt) in tqdm(self.test_loader, desc='Evaluation'):
                src = src.to(config.device)
                tgt = tgt.to(config.device)
                
                decoder_inputs= tgt[:,:-1]
                labels = tgt[:,1:]
                
                logits = self.model(src, decoder_inputs) 
                loss = self.criterion(logits.reshape(-1, logits.size(1)), labels.reshape(-1))
                total_loss += loss.item()
                batch_count += 1
                
                predictions = logits.argmax(dim=-1) 
                
                for idx in range(len(predictions)):
                   
                    prediction = [self.vi_tokenizer.convert_ids_to_tokens(token_id.item()) 
                                 for token_id in predictions[idx]]
                    reference = [self.vi_tokenizer.convert_ids_to_tokens(token_id.item()) 
                                for token_id in labels[idx]]
                    
                   
                    prediction = [token for token in prediction 
                                 if token not in ['[PADDING]', '[BOS]', '[EOS]']]
                    reference = [token for token in reference 
                               if token not in ['[PADDING]', '[BOS]', '[EOS]']]
                    
                    
                    score = bleu_score(reference, prediction)  
                    total_score += score
                    
            test_loss = total_loss / batch_count
            logger.info(f'Test Loss: {test_loss:.4f}')
            logger.info(f'Bleu score: {total_score:.2f}')
            
        return test_loss

        
        
def main():
    trainer = Trainer()
    trainer.run()


if __name__ == '__main__':
    main()
        
        
        