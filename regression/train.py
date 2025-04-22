from stock_price_prediction.preprocess_stock_data import process_apple_data
from rnn import StocksRnn
import torch.optim as optim
import torch
import torch.nn as nn

def train(model,x,y, num_epochs = 1000):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

file_path = 'RNN/apple_data.csv'
processed_df = process_apple_data(file_path)


x = torch.tensor(processed_df['Price'])
y = torch.tensor(processed_df['Price'].iloc[1:])

seq_len = 5
batch_size = 4
fetures = 1
x = x.reshape(seq_len, batch_size, fetures)
# y = torch.arange(2,12 , dtype = torch.float32)
y = y.reshape(seq_len, batch_size, fetures)
print(x)
# model = StocksRnn(1,30,1)