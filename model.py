import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


# simple CNN model for single channel 28x28 greyscale images

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(28, 28, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # Dummy pass to get the right size
        with torch.no_grad():
            dummy_output = self.features(torch.zeros(1, 1, 28, 28))
            flattened_size = dummy_output.numel() 
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        


class MNISTClassifier(pl.LightningModule):

    def __init__(self, model, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn = Accuracy(task="multiclass", num_classes=10)
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        features, true_labels = batch
        pred_labels = self(features)
        loss = self.loss_fn(pred_labels, true_labels)
        preds = torch.argmax(pred_labels, dim=1)
        accuracy = self.acc_fn(preds,true_labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)      
        return loss        

    def validation_step(self, batch, batch_idx):
        features, true_labels = batch
        pred_labels = self(features)
        loss =  self.loss_fn(pred_labels, true_labels)
        preds = torch.argmax(pred_labels, dim=1)
        accuracy = self.acc_fn(preds,true_labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
   
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate)
        return optimizer