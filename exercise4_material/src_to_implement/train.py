import torch
from data import ChallengeDataset
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from trainer import Trainer

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
csv_path = 'data.csv'
data = pd.read_csv(csv_path, sep=';')
train, test = train_test_split(data)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_data = DataLoader(ChallengeDataset(train.reset_index(drop=True), 'train'), batch_size=16, shuffle=True)
val_data = DataLoader(ChallengeDataset(test.reset_index(drop=True), 'val'), batch_size=16, shuffle=True)

# create an instance of our ResNet model
model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
early_stopping_patience = 3
trainer = Trainer(model, criterion, optimizer, train_data, val_data, True, early_stopping_patience)

# go, go, go... call fit on trainer
epochs = 100
res = trainer.fit(epochs)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='validation loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
plt.show()