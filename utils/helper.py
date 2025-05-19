from tokenizers import models, trainers, pre_tokenizers, Tokenizer
from transformers import  PreTrainedTokenizerFast
import matplotlib.pyplot as plt
import numpy as np

def create_dict(df, vocab_size=50000, min_frequency=3):
    en_tokenizer = Tokenizer(models.WordLevel(unk_token= '[UNK]'))
    vi_tokenizer = Tokenizer(models.WordLevel(unk_token='[UNK]'))

    en_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    vi_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer= trainers.WordLevelTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens =['[UNK]','[PADDING]','[BOS]','[EOS]','[SEP]'])

    en_tokenizer.train_from_iterator(df['en'], trainer)
    vi_tokenizer.train_from_iterator(df['vi'], trainer)

    en_tokenizer.save("tokenizers/en_tokenizer.json")
    vi_tokenizer.save("tokenizers/vi_tokenizer.json")


def get_tokenizers():
    en_tokenizer_file= 'tokenizers/en_tokenizer.json'
    vi_tokenizer_file= 'tokenizers/vi_tokenizer.json'
    en_tokenizer = PreTrainedTokenizerFast(tokenizer_file=en_tokenizer_file, bos_token='[BOS]',eos_token='[EOS]', pad_token='[PADDING]', unk_token='[UNK]', sep_token='[SEP]')
    vi_tokenizer = PreTrainedTokenizerFast(tokenizer_file=vi_tokenizer_file, bos_token='[BOS]',eos_token='[EOS]', pad_token='[PADDING]', unk_token='[UNK]', sep_token='[SEP]')
    
    return en_tokenizer, vi_tokenizer


def get_lr(optimizer):
    for params in optimizer.param_groups:
        return params["lr"]


def training_visualizing(history):
    assert len(history['training'])== len(history['valid']), "Training and validation loss lengths must match."
    
    epochs = range(1,len(history['training'])+1)
    
    training_loss=[sample.detach().numpy() for sample in history['training']]
    valid_loss=[sample.cpu().numpy() for sample in history['valid']]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_loss, label='Training Loss', marker='o')
    plt.plot(epochs, valid_loss, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_visualizing.png")
    plt.close()