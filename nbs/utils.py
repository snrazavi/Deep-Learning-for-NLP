def generate_bigrams(x):
    ''' Genterate all unigrams and bigrams for the input text.
    
        Inputs:
            -x: a text (str)
    '''     
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


def display_classification_result(sentence, label, label_idx):
    ''' Displays the label and the input sentence with a proper bg color.
    
        Inputs:
            - sentence: the input sentence to be classified.
            - label: predicted label of the class which sentence belongs to.
            - label_idx: index of the predicted label in the label vocab.
    '''
    print(f"{label:8s}\x1b[{41 + 2 * label_idx}m" + sentence + "\x1b[m")
    
    

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
