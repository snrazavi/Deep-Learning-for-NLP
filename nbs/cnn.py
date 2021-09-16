import torch
import torch.nn as nn
import torch.nn.functional as F


### Fully Convolutional model with attention ###
class ConvSelfAttention(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(filters, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    
    def forward(self, encoder_outputs):
        # encoder_outputs = [batch size, sent len, filters]
        energy = self.projection(encoder_outputs)
        # energy = [batch size, sent len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # weights = [batch size, sent len]
        outputs = (encoder_outputs * weights.unsqueeze(-1))
        # outputs = [batch size, sent len, filters]
        return outputs


class AttentionCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        self.attention = ConvSelfAttention(n_filters)
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #x = [sent len, batch size]
        x = x.permute(1, 0)   
        #x = [batch size, sent len]
        embedded = self.embedding(x)
        #embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]
        conved_att = [(self.attention(conv.permute(0, 2, 1))).permute(0, 2, 1) for conv in conved]
        #conved_att = [batch size, n_filters, sent len - filter sizes[i]]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_att]
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batch size, n_filters * len(filter_sizes)]
        return self.fc(cat)
