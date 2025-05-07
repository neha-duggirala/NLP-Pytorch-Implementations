import torch
import torch.nn as nn
import torch.nn.functional as F


class simpleRnn(nn.Module):
    def __init__(self, no_of_tokens, embed_dim, hid_size):
        super(simpleRnn, self).__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=no_of_tokens, embedding_dim=embed_dim
        )
        self.rnn = nn.RNN(embed_dim, hidden_size=hid_size, batch_first=True)
        self.fc = nn.Linear(hid_size, no_of_tokens)

    def forward(self, inp):
        embeddings = self.embedding_layer(inp)
        _, op = self.rnn(embeddings)
        output = self.fc(op.squeeze(0))
        # output = F.softmax(output, dim=-1)
        return output

    def __str__(self):
        return f"simpleRnn(no_of_tokens={self.embedding_layer.num_embeddings}, embed_dim={self.embedding_layer.embedding_dim}, hid_size={self.rnn.hidden_size})"


if __name__ == "__main__":
    total_voacab = 10
    embedding_vec_size = 4
    e = simpleRnn(total_voacab, embedding_vec_size, 5)
    # X = torch.tensor([4,1,3])
    X = torch.randint(size=(1, 3), low=1, high=total_voacab)
    print(e(X))
