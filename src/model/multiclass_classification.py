#!/usr/bin/env python3
""" binary_classification.py: build binary classification models
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


import torchvision
from torchvision import utils
import torchvision.transforms as transforms
from torchvision import datasets, models

import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


class BaseDataset(Dataset):
    def __init__(self, df, split, transform = None):
        super().__init__()
        self.split = df.loc[df['split'] == split].reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.split)
    
    def __getitem__(self,index):
        image = Image.open(self.split.loc[index, 'filename']).convert('RGB')
        label = self.split.loc[index, 'label']
        
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    

def load_dataset(data, batch_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
        
    data_transforms = {
        'train': transforms.Compose([transforms.RandomGrayscale(p=0.1),
                                      transforms.RandomRotation(10),
                                      #transforms.RandomResizedCrop((224, 224), scale=(0.05, 0.95), ratio=(0.9, 1.1)),
                                      #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                      #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std),
                                      ]),
        'validation': transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std),
                                   ]),
        'test': transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std),
                                    ])
    }
        
    train_data = BaseDataset(data, 'train', data_transforms['train'])
    val_data = BaseDataset(data, 'val', data_transforms['validation'])
    test_data = BaseDataset(data, 'test', data_transforms['test'])

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, drop_last=False)
        
    classes = sorted(data.label.unique().tolist())
    num_classes = len(classes)
    
    return train_data, val_data, test_data, train_loader, val_loader, test_loader, classes, num_classes



class Customized_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Customized_CNN, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=56, stride=1, padding=0)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x
    
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block

    
    
class multiclass_classification_models():
    
    def train_model(data, model_name, batch_size = 32, pretrained = True, optimizer = 'Adam', learning_rate = 0.0001, lr_decay = True, num_epochs = 25):
        
        train_data, val_data, test_data, train_loader, val_loader, test_loader, classes, num_classes = load_dataset(data, batch_size = batch_size)
        
        print('Please choose one model from the available models: Customized_CNN, Resnet18, Resnet34, Resnet50, Resnet101, Resnet152, Alexnet, VGG11, VGG13, VGG16, VGG19, Densenet121, Densenet169, Densenet201, Squeezenet1_0.')
        
        dic_label_to_index = dict()
        dic_index_to_label = dict()
        for index, label in enumerate(classes):
            dic_label_to_index[label] = index
            dic_index_to_label[index] = label
            
        if model_name == 'Customized_CNN':
            model = Customized_CNN(num_classes)
        elif model_name == 'Resnet18':
            model = models.resnet18(pretrained = pretrained)
            model.classifier = nn.Linear(1024, num_classes)
        elif model_name == 'Resnet34':
            model = models.resnet34(pretrained = pretrained)
            model.classifier = nn.Linear(1024, num_classes)           
        elif model_name == 'Resnet50':
            model = models.resnet50(pretrained = pretrained)
            model.classifier = nn.Linear(1024, num_classes)
        elif model_name == 'Resnet101':
            model = models.resnet101(pretrained = pretrained)
            model.classifier = nn.Linear(1024, num_classes)
        elif model_name == 'Resnet152':
            model = models.resnet152(pretrained = pretrained)
            model.classifier = nn.Linear(1024, num_classes)
        elif model_name == 'Alexnet':
            model = models.alexnet(pretrained = pretrained)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == 'VGG11':
            model = models.vgg11(pretrained = pretrained)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == 'VGG13':
            model = models.vgg13(pretrained = pretrained)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == 'VGG16':
            model = models.vgg16(pretrained = pretrained)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == 'VGG19':
            model = models.vgg19(pretrained = pretrained)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == 'Densenet121':
            model = models.densenet121(pretrained = pretrained)
            model.classifier = nn.Linear(1024, num_classes)
        elif model_name == 'Densenet169':
            model = models.densenet121(pretrained = pretrained)
            model.classifier = nn.Linear(1024, num_classes)
        elif model_name == 'Densenet201':
            model = models.densenet121(pretrained = pretrained)
            model.classifier = nn.Linear(1024, num_classes)
        elif model_name == 'Squeezenet1_0':
            model = models.squeezenet1_0(pretrained = pretrained)
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
    
        if optimizer == 'SGD':
            optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer == 'Adam':
            optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate)
    
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

        best_model_wts = copy.deepcopy(model.state_dict())
        val_best_acc = 0.0
        
        
        accuracy_stats = {
            'train': [],
            'val': []
        }
        loss_stats = {
            'train': [],
            'val': []
        }

        print("Begin training.")
        for epoch in range(1, num_epochs+1):
            ###### Model training
            model.train()  # Set model to training mode
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = torch.as_tensor([dic_label_to_index[label] for label in labels])  ## from original labels to label-index
                labels = labels.to(device)
                optimizer_ft.zero_grad()
                
                if model_name != 'Customized_CNN':
                    outputs = model(inputs)
                elif model_name == 'Customized_CNN':
                    outputs = model(inputs).squeeze()
                    
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_ft.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).cpu().numpy()
                if lr_decay == True:
                        exp_lr_scheduler.step()
            train_epoch_loss = running_loss / len(train_data)
            train_epoch_acc = running_corrects / len(train_data)
            
            ###### Model evaluation
            model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = torch.as_tensor([dic_label_to_index[label] for label in labels])  ## from original labels to label-index
                labels = labels.to(device)
                optimizer_ft.zero_grad()
                
                if model_name != 'Customized_CNN':
                    outputs = model(inputs)
                elif model_name == 'Customized_CNN':
                    outputs = model(inputs).squeeze()
                    
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).cpu().numpy()
            val_epoch_loss = running_loss / len(val_data)
            val_epoch_acc = running_corrects / len(val_data)
            if val_epoch_acc > val_best_acc:
                val_best_acc = val_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            ###### results summary for each epoch
            loss_stats['train'].append(train_epoch_loss)
            loss_stats['val'].append(val_epoch_loss)
            accuracy_stats['train'].append(train_epoch_acc)
            accuracy_stats['val'].append(val_epoch_acc)
            print(f'Epoch {epoch+0:02}: | Train Loss: {train_epoch_loss:.5f} | Val Loss: {val_epoch_loss:.5f} | Train Acc: {train_epoch_acc:.3f}| Val Acc: {val_epoch_acc:.3f}')
    
        print('Best validation accuracy: {:4f}'.format(val_best_acc))
        # load best model weights
        model.load_state_dict(best_model_wts)
        
        train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
        sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')
        plt.show()
        
        ###### Model scoring on testset
        y_pred = []
        y_true = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = torch.as_tensor([dic_label_to_index[label] for label in labels])  ## from original labels to label-index
            labels = labels.to(device)
            if model_name != 'Customized_CNN':
                outputs = model(inputs)
            elif model_name == 'Customized_CNN':
                outputs = model(inputs).squeeze()
            _, preds = torch.max(outputs, 1)
            y_pred += preds.cpu().tolist()
            y_true  += labels.cpu().tolist()
    
        print('Scoring results on testset:')
        print(classification_report(y_true, y_pred))
        print('Confusion matrix on testset:')
        cm = confusion_matrix([dic_index_to_label[x] for x in y_true], [dic_index_to_label[x] for x in y_pred], classes)
        print(pd.DataFrame(cm, index=['true: {:}'.format(x) for x in classes], columns=['pred: {:}'.format(x) for x in classes]))
        
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Blues')
        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(classes); 
        ax.yaxis.set_ticklabels(classes);

        return model    
    
    
def score_NewData():
    """ TODO """