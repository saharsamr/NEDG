def save_predictions(input, label, prediction):

    with open('preds.csv', 'w') as f:
        for i, l, p in zip(input, label, prediction):
            f.write(f'{i}||{l}||{p}\n')
