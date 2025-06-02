import torch
import torch.nn as nn



# class SimpleLSTM(nn.Module):
#     def __init__(self, vocab_size,embed_dim):
#         super(SimpleLSTM,self).__init__()
#         self.embedding_layer = nn.Embedding(vocab_size,embed_dim)
#         self.lstm = nn.LSTM(embed_dim,150,batch_first = True)
#         self.fc = nn.Linear(150,vocab_size)

#     def forward(self,x):
#         vec_embedding = self.embedding_layer(x)
#         all_hidden_states, (output, cell_state) = self.lstm(vec_embedding)
#         op = self.fc(output.squeeze(0))
#         return op

class SimpleLSTM(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, 100)
    self.lstm = nn.LSTM(100, 150, batch_first=True)
    self.fc = nn.Linear(150, vocab_size)

  def forward(self, x):
    embedded = self.embedding(x)
    intermediate_hidden_states, (final_hidden_state, final_cell_state) = self.lstm(embedded)
    output = self.fc(final_hidden_state.squeeze(0))
    return output