import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttention(nn.Module):

    def __init__(self, d_model=2, r_dim=0, c_dim=1):
        super().__init__()
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.r_dim = r_dim
        self.c_dim = c_dim

    def forward(self, token_encodings, mask=False):
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        #unscaled cosine similaries
        sims = torch.matmul(q,k.transpose(dim0=self.r_dim, dim1=self.c_dim))

         ## Scale the similarities by dividing by sqrt(k.c_dim)
        scaled_sims = sims / torch.tensor(k.size(self.c_dim)**0.5)

        if mask is not None:
            scaled_sims =scaled_sims.masked_fill(mask=mask, value=-1e9) # I've also seen -1e20 and -9e15 used in masking

        ## Apply softmax to determine what percent of each tokens' value to
        ## use in the final attention values.
        attention_percents = F.softmax(scaled_sims, dim=self.c_dim)

        ## Scale the values by their associated percentages and add them up.
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
