import pandas as pd
import numpy as np
import torch


def data_cleaning(df):
    df['en_len']= df['en'].apply(lambda x: len(x.split(" ")))

    min_threshold = df['en_len'].quantile(0.05)
    max_threshold = df['en_len'].quantile(0.99)
    return df[(df['en_len']>=min_threshold) & (df['en_len']<= max_threshold)]


def preprocess(df , seq_len, en_tokenizer, vi_tokenizer ):
    
    src_text= list(df['en'])
    tgt_text = ['[BOS]'+ " "+ text+" "+'[EOS]' for text in df['vi'] ]
    
    src_input_ids = en_tokenizer(src_text,truncation=True,max_length=seq_len,padding='max_length',return_tensors='pt')['input_ids']
    tgt_input_ids = vi_tokenizer(tgt_text,truncation=True,max_length=seq_len,padding='max_length',return_tensors='pt')['input_ids']
    
    return {
        'input_ids': src_input_ids,
        'labels': tgt_input_ids
    }
 
def punctuation_handling(sentence):
    new_sentence = ""
    
    for idx,character in enumerate(sentence):
        if character in [',','.','-',"'","?","!",":","{","}","(",")"]: 
            if len(new_sentence)!=0 and new_sentence[-1]!=" ":
                new_sentence+= " "
            new_sentence+=character
            new_sentence+=" "
        else:
            if character == " ":
                if len(new_sentence)!=0 and new_sentence[-1]==" ":
                    continue
                else:
                    new_sentence+= character
            else:
                new_sentence+=character

    new_sentence= new_sentence.strip().lower()
    
    return new_sentence
    


def data_handling(df):
    df['en']= df['en'].apply(lambda x: punctuation_handling(x))
    df['vi']= df['vi'].apply(lambda x: punctuation_handling(x))

    return df    