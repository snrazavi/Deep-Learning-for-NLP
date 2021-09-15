from tqdm import tqdm
import time
import torch
import pandas as pd

from IPython.display import clear_output, display


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# A helper function to convert training time for each epoch to minutes and seconds 
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    pbar = tqdm(iterator, position=0, desc=f'Training:   | Loss={0:.4f} | Acc={0:.4f} |')
    
    for batch in pbar:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text)
        loss = criterion(predictions, batch.label)
        acc = categorical_accuracy(predictions, batch.label)
        pbar.set_description(f'Training:   | Loss={loss:.4f} | Acc={acc:.4f} |')
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    pbar = tqdm(iterator, position=0, desc=f'Validation: | Loss={0:.4f} | Acc={0:.4f} |')
    with torch.no_grad():
    
        for batch in pbar:

            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            pbar.set_description(f'Validation: | Loss={loss:.4f} | Acc={acc:.4f} |')

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_model(model, device, train_iterator, valid_iterator, optimizer, criterion, scheduler, n_epochs, fname):

    # Create a dataframe to report train stats
    report_df = pd.DataFrame(
        columns=['Epoch', 'Train Loss', 'Valid Loss', 'Train Acc', 'Valid Acc', 'Time']
    )

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):

        if epoch > 0:
            clear_output(True)
        display(report_df)

        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        scheduler.step()
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if a better model is found, replace current model with the better one
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), fname)

        report_df.loc[epoch] = [
            f'{epoch + 1}',
            f'{train_loss:.3f}', f'{valid_loss:.3f}',
            f'{train_acc*100:.2f}', f'{valid_acc*100:.2f}',
            f'{epoch_mins}m {epoch_secs}s'
        ]

    clear_output(wait=True)
    display(report_df)
    model.load_state_dict(torch.load(fname, map_location=device))
