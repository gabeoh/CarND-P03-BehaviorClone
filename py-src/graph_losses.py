import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

losses_model1 = [
    [0.0293, 0.0212, 0.0181, 0.0156, 0.0135, 0.0120],
    [0.0232, 0.0202, 0.0179, 0.0171, 0.0167, 0.0174]
]

losses_model2 = [
    [0.0390, 0.0310, 0.0281, 0.0268, 0.0256, 0.0249, 0.0241, 0.0240, 0.0232, 0.0226, 0.0216, 0.0212, 0.0207, 0.0199,
     0.0201, 0.0194, 0.0180, 0.0189, 0.0175, 0.0174, 0.0172, 0.0169, 0.0164, 0.0159, 0.0160, 0.0155, 0.0150, 0.0150,
     0.0140, 0.0142],
    [0.0297, 0.0270, 0.0221, 0.0221, 0.0265, 0.0244, 0.0220, 0.0216, 0.0191, 0.0199, 0.0195, 0.0192, 0.0195, 0.0172,
     0.0186, 0.0173, 0.0164, 0.0178, 0.0185, 0.0170, 0.0179, 0.0185, 0.0159, 0.0164, 0.0174, 0.0158, 0.0179, 0.0161,
     0.0160, 0.0163]
]

def graph_losses(loesses, title=None, outfile=None):
    train_loss = np.array(loesses[0])
    valid_loss = np.array(loesses[1])
    assert len(train_loss) == len(valid_loss), "The training and validation losses should have the same length"
    epochs = np.array(range(len(train_loss))) + 1

    df_losses = pd.DataFrame({
        'Train Loss': train_loss,
        'Validation Loss': valid_loss
    }, index=epochs, columns=['Train Loss', 'Validation Loss'])
    print(df_losses)
    df_losses.plot()
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.ylim(ymax=0.04)
    if (title):
        plt.title(title)
    if (outfile):
        plt.savefig(outfile)
    plt.show()

graph_losses(losses_model1, 'Model Losses', 'model_losses.png')
graph_losses(losses_model2, 'Model Losses W/ Dropout', 'model_losses_dropout.png')
