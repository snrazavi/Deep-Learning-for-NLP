import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data

from argparse import ArgumentParser
from transformers import BertTokenizerFast, BertModel

from bert import BERTClassifier
from training import train_model, evaluate


def get_args():
    parser = ArgumentParser(description="Bert")
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0) # Use -1 for CPU
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--vocab_size', type=int, default=25000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='SST-1')
    parser.add_argument('--resume_snapshot', type=str, default=None)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--save_path', type=str, default='saves')
    parser.add_argument('--output_channel', type=int, default=100)
    parser.add_argument('--words_dim', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--epoch_decay', type=int, default=1)
    parser.add_argument('--trained_model', type=str, default="")
    parser.add_argument('--weight_decay',type=float, default=0)
    parser.add_argument('--hidden_dim',type=int, default=256)
    parser.add_argument('--num_layers',type=int, default=1)

    args = parser.parse_args()
    return args


# BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens


if __name__=="__main__":

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    args = get_args()

    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id

    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

    # VOCABS
    TEXT = data.Field(batch_first = True,
                      use_vocab = False,
                      tokenize = tokenize_and_cut,
                      preprocessing = tokenizer.convert_tokens_to_ids,
                      init_token = init_token_idx,
                      eos_token = eos_token_idx,
                      pad_token = pad_token_idx,
                      unk_token = unk_token_idx)

    LABEL = data.LabelField()

    # load train, validation and test data from corresponding CSV files
    print(f'[INFO] Loading datasets...')
    train_data, valid_data = data.TabularDataset.splits(
        path='../data', 
        train='spam-train.csv',
        validation='spam-valid.csv',
        format='csv', 
        skip_header=True,
        fields=[('label', LABEL),('text', TEXT)]
    )

    # Build vocabulary for labels
    LABEL.build_vocab(train_data)

    # Define iterators to iterate through different datasets 
    # during training and testing model

    BATCH_SIZE = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu == 0 else 'cpu')
    print(f'[INFO] Using {device}...')


    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text),
        device=device)

    bert = BertModel.from_pretrained('bert-base-uncased')

    # model hyper-parameters
    HIDDEN_DIM = args.hidden_dim
    OUTPUT_DIM = len(LABEL.vocab)
    N_LAYERS = args.num_layers
    BIDIRECTIONAL = True
    DROPOUT = args.dropout

    # Create the BERT model
    model = BERTClassifier(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

    # freeze all bert parameters
    for name, param in model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = False

    #################################
    ### DEFINE LOSS AND OPTIMIZER ###
    #################################
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # move the model and the loss function to GPU if there is a GPU 
    model = model.to(device)
    criterion = criterion.to(device)

    #############
    ### TRAIN ###
    #############
    print(f'[INFO] Strat training...')
    fname=f'models/bert-{args.num_layers}-{args.hidden_dim}.pt'

    train_model(
        model, 
        device,
        train_iterator, 
        valid_iterator, 
        optimizer, 
        criterion, 
        scheduler, 
        n_epochs=args.epochs, 
        fname=fname)

    model.load_state_dict(torch.load(fname, map_location=device))

    print(f'[INFO] Testing on test dataset...')
    test_loss, test_acc = evaluate(model, valid_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')









    




