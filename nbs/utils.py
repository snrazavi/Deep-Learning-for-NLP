def display_classification_result(sentence, label, label_idx):
    ''' Displays the label and the input sentence with a proper bg color.
    
        Inputs:
            - sentence: the input sentence to be classified.
            - label: predicted label of the class which sentence belongs to.
            - label_idx: index of the predicted label in the label vocab.
    '''
    print(f"{label:8s}\x1b[{41 + 2 * label_idx}m" + sentence + "\x1b[m")