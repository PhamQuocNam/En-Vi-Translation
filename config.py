import torch

class Config:
    """Configuration for T5 model training and inference"""
    
    # Paths and files
    data_path = "en-vi-dataset"
    checkpoint_dir = "checkpoints"
    output_dir = "outputs"
    tensorboard_dir = "runs"
    log_file = "training.log"
    
    # Model configuration
    model_name = "T5Model"
    model_type = "t5-base"  
    pretrained_weight = "checkpoints/pretrained.pth"
    
    # Model architecture
    embedding_dim = 512    
    num_layers = 4        
    n_heads = 4           
    d_ff = 2048             
    dropout = 0.2         
    
    # Tokenizer settings
    tokenizer_dict = 'tokenizers'
    vocab_size = 50000     
    max_seq_length = 40    
    padding_idx= 1
    
    # Training hyperparameters
    batch_size = 64
    lr = 0.0001    
    weight_decay = 0.01     
    epochs = 10
    
    # Optimization
    optimizer = "Adam"     
    factor= 0.1
    patience = 5
    
    # Evaluation and generation
    eval_steps = 1000      
    save_steps = 5000      
    generation_max_length = 128
    generation_num_beams = 4
    
    
    # Miscellaneous
    seed = 42              
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    debug = False           
    