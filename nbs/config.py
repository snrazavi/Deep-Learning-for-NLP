class RCNNConfig(object):
    embed_size = 100
    hidden_layers = 2
    hidden_size = 64
    output_size = 2
    max_epochs = 5
    hidden_size_linear = 64
    lr = 0.5
    batch_size = 128
    seq_len = None # Sequence length for RNN
    dropout_keep = 0.8