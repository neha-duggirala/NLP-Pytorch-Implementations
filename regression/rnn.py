import torch
import torch.nn as nn


class StocksRnn(nn.Module):
    def __init__(self, input_dim, hidden_states, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim,hidden_states, batch_first=True)
        self.FC = nn.Linear(hidden_states,output_dim)
        

    def forward(self,x):
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size)
        out,_ = self.rnn(x,h0)
        out = self.FC(out)
        return out

# x = torch.arange(1,11 , dtype = torch.float32)

# seq_len = 10
# batch_size = 2
# fetures = 1
# x = x.reshape(seq_len, batch_size, fetures)
# y = torch.arange(2,12 , dtype = torch.float32)
# y = y.reshape(seq_len, batch_size, fetures)
# print(x)
# model = StocksRnn(1,30,1)
# print(model(x))
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 1000
# for epoch in range(num_epochs):
#     model.train()
#     outputs = model(x)
#     loss = criterion(outputs, y)
    
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# print(outputs,y)
