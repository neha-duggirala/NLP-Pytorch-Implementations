
import torch
from masked_attention import MaskedAttention
from self_attention import SelfAttention
from utils import visualize_tensor
from word_embeddings import word_vectors, words



self_attention = SelfAttention(d_model=2,
                               r_dim=0,
                               c_dim=1)

self_attention_values = self_attention(word_vectors)


maskedSelfAttention = MaskedAttention(d_model=2,
                               r_dim=0,
                               c_dim=1)


mask = torch.tril(torch.ones(len(word_vectors), len(word_vectors)))
mask = mask == 0

masked_attention_values = maskedSelfAttention(word_vectors,mask)

visualize_tensor(self_attention_values, words)
visualize_tensor(masked_attention_values, words)