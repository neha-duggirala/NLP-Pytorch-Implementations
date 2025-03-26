import torch
import torch.nn as nn
from collections import Counter

# This is going to be the dummy sentence :
document = "Neha ate Pizza and she is happy"
 
words = document.split(' ')
 
# create a dictionary
vocab = Counter(words) 
vocab = sorted(vocab, key=vocab.get, reverse=True)
vocab_size = len(vocab)
 
# create a word to index dictionary from our Vocab dictionary
word2idx = {word: ind for ind, word in enumerate(vocab)} 
 
encoded_sentences = [word2idx[word] for word in words]
 
# assign a value to your embedding_dim
e_dim = 2

# initialise an Embedding layer from Torch
emb = nn.Embedding(vocab_size, e_dim)
word_vectors = emb(torch.LongTensor(encoded_sentences))
 
#print the word_vectors
print(document,word_vectors)