import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

PARAMS = {'batch_size': 64,
          'linear': 256,
          'lr': 0.007,
          'decay_factor': 0.9,
          'max_epochs': 7}


class LitModel(pl.LightningModule):
    def __init__(self, linear, learning_rate, decay_factor):
        super().__init__()
        self.linear = linear
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.train_img_max = 10
        self.train_img = 0
        self.layer_1 = torch.nn.Linear(28 * 28, linear)
        self.layer_2 = torch.nn.Linear(linear, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = LambdaLR(optimizer, lambda epoch: self.decay_factor ** epoch)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=False)
        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        return {'loss': loss,
                'y_true': y_true,
                'y_pred': y_pred}

    def training_epoch_end(self, outputs):
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])
        acc = accuracy_score(y_true, y_pred)
        self.log('train_acc', acc)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=False)
        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        return {'loss': loss,
                'y_true': y_true,
                'y_pred': y_pred}

    def validation_epoch_end(self, outputs):
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])
        acc = accuracy_score(y_true, y_pred)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, prog_bar=False)
        y_true = y.cpu().detach().numpy()
        y_pred = y_hat.argmax(axis=1).cpu().detach().numpy()
        for j in np.where(np.not_equal(y_true, y_pred))[0]:
            img = np.squeeze(x[j].cpu().detach().numpy())
            img[img < 0] = 0
            img = (img / img.max()) * 256
            neptune_logger.experiment.log_image('test_misclassified_images',
                                                img,
                                                description='y_pred={}, y_true={}'.format(y_pred[j], y_true[j]))
        return {'loss': loss,
                'y_true': y_true,
                'y_pred': y_pred}

    def test_epoch_end(self, outputs):
        y_true = np.array([])
        y_pred = np.array([])
        for results_dict in outputs:
            y_true = np.append(y_true, results_dict['y_true'])
            y_pred = np.append(y_pred, results_dict['y_pred'])
        acc = accuracy_score(y_true, y_pred)
        self.log('test_acc', acc)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, normalization_vector):
        super().__init__()
        self.batch_size = batch_size
        self.normalization_vector = normalization_vector
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True)
        MNIST(os.getcwd(), train=False, download=True)

    def setup(self, stage):
        # transforms
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(self.normalization_vector[0],
                                                             self.normalization_vector[1])])
        if stage == 'fit':
            mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == 'test':
            self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
        return mnist_test


def log_confusion_matrix(lit_model, data_module):
    lit_model.freeze()
    test_data = data_module.test_dataloader()
    y_true = np.array([])
    y_pred = np.array([])
    for i, (x, y) in enumerate(test_data):
        y = y.cpu().detach().numpy()
        y_hat = lit_model.forward(x).argmax(axis=1).cpu().detach().numpy()
        y_true = np.append(y_true, y)
        y_pred = np.append(y_pred, y_hat)

    fig, ax = plt.subplots(figsize=(16, 12))
    plot_confusion_matrix(y_true, y_pred, ax=ax)
    neptune_logger.experiment.log_image('confusion_matrix', fig)


lr_logger = LearningRateMonitor(logging_interval='epoch')

model_checkpoint = ModelCheckpoint(filepath='my_model/checkpoints/{epoch:02d}-{val_loss:.2f}',
                                   save_weights_only=True,
                                   save_top_k=3,
                                   monitor='val_loss',
                                   period=1)

neptune_logger = NeptuneLogger(project_name='neptune-ai/tour-with-pytorch',
                               close_after_fit=False,
                               experiment_name='pytorch-lightning',
                               params=PARAMS,
                               tags=['pytorch-lightning', 'MNIST'])

trainer = pl.Trainer(logger=neptune_logger,
                     callbacks=[lr_logger, model_checkpoint],
                     log_every_n_steps=100,
                     max_epochs=PARAMS['max_epochs'],
                     track_grad_norm=2)

model = LitModel(linear=PARAMS['linear'],
                 learning_rate=PARAMS['lr'],
                 decay_factor=PARAMS['decay_factor'])

dm = MNISTDataModule(normalization_vector=((0.1307,), (0.3081,)),
                     batch_size=PARAMS['batch_size'])

trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)

# Log model checkpoints to Neptune
for k in model_checkpoint.best_k_models.keys():
    model_name = 'checkpoints/' + k.split('/')[-1]
    neptune_logger.experiment.log_artifact(k, model_name)

# Log best model checkpoint score to Neptune
neptune_logger.experiment.set_property('best_model_score', model_checkpoint.best_model_score.tolist())

# Log confusion matrix
log_confusion_matrix(model, dm)

# Stop Neptune logger at the end
neptune_logger.experiment.stop()
