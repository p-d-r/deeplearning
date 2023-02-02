import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import torch
import numpy as np


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        if cuda:
            self._model = model.to(device)
            self._crit = crit.to(device)

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # train by GPU
        if self._cuda:
            x = x.to(device)
            y = y.to(device)

        # reset parameters after one time
        self._model.zero_grad()

        # calculate through forward network
        output = self._model.forward(x)

        # define loss
        loss = self._crit(output, y.detach())

        # Backpropagation
        loss.backward()

        # optimizer
        self._optim.step()
        return loss

    def val_test_step(self, x, y):

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # test by GPU
        if self._cuda:
            x = x.to(device)
            y = y.to(device)

        # calculate through forward network
        y_prediction = self._model.forward(x).round()

        # calculate test loss
        loss = self._crit(y_prediction, y.float())
        return loss, y_prediction

    def train_epoch(self):

        # define average loss
        loss = 0
        self._model.train()

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # train by GPU
        if self._cuda:
            self._model = self._model.to(device)
            self._crit = self._crit.to(device)

        # calculate loss with train dataset
        for image, label in tqdm(self._train_dl, desc='train'):
            image = image.requires_grad_(True)
            label = label.float().requires_grad_(True)
            loss += Trainer.train_step(self, image, label).item()

        # calculate average loss
        loss = loss / len(self._train_dl)

        return loss

    def val_test(self):
        # start test eval model
        self._model.eval()

        # define loss pre and labels
        loss_test = 0
        predicts = []
        labels = []

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # calculate loss with test dataset
        for image, label in tqdm(self._val_test_dl, desc='val'):
            image = image.requires_grad_(False)
            label = label.requires_grad_(False)

            if self._cuda:
                image = image.to(device)
                label = label.to(device)
            labels.append(label.cpu().tolist())
            loss, y_prediction = Trainer.val_test_step(self, image, label)
            loss_test += loss.item()
            predicts.append(y_prediction.cpu().tolist())
            # print(f"Using device: {device}")

        # calculate average loss
        loss_test = loss_test / len(self._val_test_dl)
        return loss_test, predicts, labels

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        # create a list for the train and validation losses, and create a counter for the epoch
        train_loss = []
        val_loss = []
        f1_list = []
        epoch = 0
        f1 = 0
        minimum = float('inf')
        finish_patience = 0
        while True:
            print('epoch ' + str(epoch))

            # calculate train and val_test loss
            t_loss = Trainer.train_epoch(self)
            v_loss, prediction, labels = Trainer.val_test(self)

            # get labels and predictions
            labels = [item for sublist in labels for item in sublist]
            prediction = [item for sublist in prediction for item in sublist]

            # calculate F1-score
            f1score = f1_score(np.around(np.array(labels).flatten()),
                               np.around(np.array(prediction).flatten()))
            f1_list.append(f1score)
            if v_loss < minimum or f1score > f1:
                minimum = v_loss
                finish_patience = 0
            else:
                finish_patience += 1

            # F1-score > 0.6, save model parameters
            if (f1score > f1) and (f1score > 0.6):
                Trainer.save_onnx(self, 'checkpoint_test.onnx')
                f1 = f1score
                Trainer.save_checkpoint(self, epoch + 1)
            print('')
            print('f1 score: ' + str(f1score))
            print('training loss: ' + str(t_loss))
            print('validation loss: ' + str(v_loss))
            print('————————————————————————————————————')

            # finish one epoch, save loss results of train and test
            epoch += 1
            train_loss.append(t_loss)
            val_loss.append(v_loss)

            # finish training conditions
            if epoch == epochs or finish_patience >= self._early_stopping_patience:
                print('f1_max: ' + str(f1))
                return train_loss, val_loss, f1_list