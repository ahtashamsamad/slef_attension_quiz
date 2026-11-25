import torch                                     # improting pytorch
import torch.nn as nn                            # importing pytorch neural network module
from torchtyping import TensorType               # imprting torchtyping 

class SingleHeadAttention(nn.Module):            # making a class for single head attention which inherits nn.Module
  """Single attention head"""

  def __init__(self, embedding_dim: int, attention_dim: int):   # initializing the class with embedding dimension and attention dimension
    super().__init__()                                          # calling the constructor of the parent class nn.Module
    torch.manual_seed(0)                                        # setting the random seed for reproducibility

    # Initialising weights
    self.wk = nn.Linear(embedding_dim, attention_dim, bias=False) # embedding_dim is the dimension of the input embeddings, attention_dim is the dimension of the attention space. This linear layer will project the input embeddings into the key space.
    self.wq = nn.Linear(embedding_dim, attention_dim, bias=False) # embedding_dim is the dimension of the input embeddings, attention_dim is the dimension of the attention space. This linear layer will project the input embeddings into the query space.
    self.wv = nn.Linear(embedding_dim, attention_dim, bias=False) # embedding_dim is the dimension of the input embeddings, attention_dim is the dimension of the attention space. This linear layer will project the input embeddings into the value space.

  def forward(self, embedded: TensorType[float]) -> TensorType[float]: # defining the forward pass of the single head attention
    # calculating Query, Key and Value
    q = self.wq(embedded) # embedded is of shape [num_sentence, num_words, embedding_dim], q will be of shape [num_sentence, num_words, attn_dim]
    k = self.wk(embedded) # 
    v = self.wv(embedded) # 
    
    
    attn_score = q @ torch.transpose(k, 1, 2) / (k.shape[-1] ** 0.5) # [batch_size, num_words, num_words] # calculating attention scores

    upper_triangular  = torch.triu(attn_score, diagonal=1).bool()     # below 2 lines is for masking in decoder block, comment it for encoder block

    attn_score[upper_triangular] = float("-inf") 

    # applying softmax
    attn_score_softmax = nn.functional.softmax(attn_score, dim = -1) # [batch_size, num_words, num_words]

    # getting weighted values by multiplying softmax of attention score with values
    weighted_values = attn_score_softmax @ v # [batch_size, num_words, attention_dim]

    return weighted_values # [batch_size, num_words, attention_dim]
