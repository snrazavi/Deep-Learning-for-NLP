import torch
import torch.nn as nn
import torch.nn.functional as F


class RCNN(nn.Module):
    
    def __init__(self, config, vocab_size):
        super(RCNN, self).__init__()
        self.config = config
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, self.config.embed_size)
        
        # Bi-directional LSTM for RCNN
        self.lstm = nn.LSTM(input_size = self.config.embed_size,
                            hidden_size = self.config.hidden_size,
                            num_layers = self.config.hidden_layers,
                            dropout = self.config.dropout_keep if self.config.hidden_layers > 1 else 0,
                            bidirectional = True)
        
        self.dropout = nn.Dropout(self.config.dropout_keep)
        
        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(
            self.config.embed_size + 2*self.config.hidden_size,
            self.config.hidden_size_linear
        )
        
        # Tanh non-linearity
        self.tanh = nn.Tanh()
        
        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.hidden_size_linear,
            self.config.output_size
        )
            
    def forward(self, x):
        # x.shape = (seq_len, batch_size)
        embedded_sent = self.embedding(x)
        # embedded_sent.shape = (seq_len, batch_size, embed_size)

        lstm_out, (h_n,c_n) = self.lstm(embedded_sent)
        # lstm_out.shape = (seq_len, batch_size, 2 * hidden_size)
        
        input_features = torch.cat([lstm_out,embedded_sent], 2).permute(1,0,2)
        # final_features.shape = (batch_size, seq_len, embed_size + 2*hidden_size)
        
        linear_output = self.tanh(self.W(input_features))
        # linear_output.shape = (batch_size, seq_len, hidden_size_linear)
        
        linear_output = linear_output.permute(0,2,1) # Reshaping fot max_pool
        
        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(2)
        # max_out_features.shape = (batch_size, hidden_size_linear)
        
        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)
        
        return final_out