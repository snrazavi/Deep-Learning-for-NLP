import torch
import tqdm
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from IPython.display import clear_output


def display_classification_result(sentence, label, label_idx):
    ''' Displays the label and the input sentence with a proper bg color.
    
        Inputs:
            - sentence: the input sentence to be classified.
            - label: predicted label of the class which sentence belongs to.
            - label_idx: index of the predicted label in the label vocab.
    '''
    print(f"{label:8s}\x1b[{41 + 2 * label_idx}m" + sentence + "\x1b[m")
    
    
def compute_confusion_matrix(model, iterator):
    """ Compute confusion matrix.

        Inputs:
            - model: a pretrained model
            - iterator: a torchtext iterator that iterates over a torchtext dataset

        Outputs:
            - Confusion matrix
    """    
    model.eval()
    
    true_labels, pred_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(iterator):
            prediction = model(batch.text)
            true_labels += [label.item() for label in batch.label]
            pred_labels += [label.item() for label in prediction.argmax(1)]
    
    clear_output(True)
    return confusion_matrix(true_labels, pred_labels, normalize='true'), true_labels, pred_labels


def plot_confusion_matrix(cm, labels):
    """ Plot the confusion matrix

        Inputs:
            - cm: a confusion matrix (numpy array)
            - labels: class labels to be shown on the vertical and horizontal axes of confusion matrix
    """
    df_cm = pd.DataFrame(cm*100, index=labels, columns=labels)
    plt.figure(figsize = (10,7))
    plt.title('Confusion Matrix')
    sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    plt.show()   
