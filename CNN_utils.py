import os, json, sys
import cv2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG,HTML
import collections
# import time
torchvision.__version__

# from sklearn.model_selection import KFold

import torch as pt
from torch.autograd import backward, grad
from torch.optim import Adam, SGD
from torch.nn.functional import relu, softmax, gumbel_softmax, binary_cross_entropy, mse_loss

from tqdm import tqdm, trange

import random

def print_data(data_path_origin, folder_to_class):
    # Data Path
    data_path = data_path_origin

    # Visualising Data
    classes = []
    img_classes = []
    n_image = []
    height = []
    width = []
    dim = []

    # Using folder names to identify classes
    for folder in os.listdir(data_path):
        classes.append(folder_to_class[folder])

        # Number of each image
        images = os.listdir(data_path+folder)
        n_image.append(len(images))

        for i in images:
            img_classes.append(folder_to_class[folder])
            img = np.array(Image.open(data_path+folder+'/'+i))
            height.append(img.shape[0])
            width.append(img.shape[1])
        dim.append(img.shape[2])

    df_train = pd.DataFrame({
        'classes': classes,
        'number': n_image,
        "dim": dim
    })
    print("heights:" + str(height[10]))
    print("Widths:" + str(width[10]))
    print(df_train)

    image_df = pd.DataFrame({
        "classes": img_classes,
        "height": height,
        "width": width
    })
    img_df_train = image_df.groupby("classes").describe()
    return img_df_train

def imshow_tensor(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image

def train(model, criterion, optimizer, train_loader, val_loader, save_location, early_stop=3, n_epochs=20, print_every=2):
   #Initializing some variables
  valid_loss_min = np.Inf
  stop_count = 0
  valid_max_acc = 0
  history = []
  model.epochs = 0
  
  #Loop starts here
  for epoch in trange(n_epochs):
    
    train_loss = 0
    valid_loss = 0
    
    train_acc = 0
    valid_acc = 0
    
    model.train()
    ii = 0
    
    for data, label in train_loader:
      ii += 1
      data, label = data.cuda(), label.cuda()
      optimizer.zero_grad()
      output = model(data)
      
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()
      
      # Track train loss by multiplying average loss by number of examples in batch
      train_loss += loss.item() * data.size(0)
      
      # Calculate accuracy by finding max log probability
      _, pred = torch.max(output, dim=1) # first output gives the max value in the row(not what we want), second output gives index of the highest val
      correct_tensor = pred.eq(label.data.view_as(pred)) # using the index of the predicted outcome above, torch.eq() will check prediction index against label index to see if prediction is correct(returns 1 if correct, 0 if not)
      accuracy = torch.mean(correct_tensor.type(torch.FloatTensor)) #tensor must be float to calc average
      train_acc += accuracy.item() * data.size(0)
      if ii%15 == 0:
        print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete.')
      
    model.epochs += 1
    with torch.no_grad():
      model.eval()
      
      for data, label in val_loader:
        data, label = data.cuda(), label.cuda()
        
        output = model(data)
        loss = criterion(output, label)
        valid_loss += loss.item() * data.size(0)
        
        _, pred = torch.max(output, dim=1)
        correct_tensor = pred.eq(label.data.view_as(pred))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
        valid_acc += accuracy.item() * data.size(0)
        
      train_loss = train_loss / len(train_loader.dataset)
      valid_loss = valid_loss / len(val_loader.dataset)
      
      train_acc = train_acc / len(train_loader.dataset)
      valid_acc = valid_acc / len(val_loader.dataset)
      
      history.append([train_loss, valid_loss, train_acc, valid_acc])
      
      if (epoch + 1) % print_every == 0:
        print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
        print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')
        
      if valid_loss < valid_loss_min:
        torch.save(model.state_dict(), save_location)
        stop_count = 0
        valid_loss_min = valid_loss
        valid_best_acc = valid_acc
        best_epoch = epoch
        
      else:
        stop_count += 1
        
        # Below is the case where we handle the early stop case
        if stop_count >= early_stop:
          print(f'\nEarly Stopping Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
          model.load_state_dict(torch.load(save_location))
          model.optimizer = optimizer
          history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc','valid_acc'])
          return model, history
        
  model.optimizer = optimizer
  print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
  
  history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
  return model, history


def evaluation(model, data_loader, criterion, classes):
    class_len     = len(classes)
    class_correct = list(0. for i in range(class_len))
    class_total = list(0. for i in range(class_len))
    
    model.eval()

    for inputs, labels in data_loader:
        with torch.no_grad():
            outputs = model(inputs.cuda())
            
            _, predicted = torch.max(outputs.data, 1)
            
            correct = np.squeeze(predicted.eq(labels.cuda().data.view_as(predicted)))

            for i in range(len(labels)):
                label = labels.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        del correct

    for i in range(class_len):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))


    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

def rule_evaluation(rule_list, x_list, y_list, hidden_sample_list, classes):
    class_len     = len(classes)
    class_correct = list(0. for i in range(class_len))
    class_total = list(0. for i in range(class_len))
    total_loss  = 0.
    for rule in tqdm(rule_list):
        local_class_correct = list(0. for i in range(class_len))
        local_class_total = list(0. for i in range(class_len))
        linear_model = rule[2]
        mask = rule[1]
        with torch.no_grad():
            print("Number of sample is",sum(mask).item())
            outputs = linear_model(hidden_sample_list[mask].cuda())
            _, predicted = torch.max(outputs.data, 1)
    #         _, preds_by_original = torch.max(pred_list[mask].cuda(),1)
            _, preds_by_original = torch.max(y_list[mask].cuda(),1)

    #         print(predicted,preds_by_original)
            correct = (predicted == preds_by_original)
            for i in range(len(predicted)):
                label = preds_by_original.data[i]
                local_class_correct[label] += correct[i].item()
                local_class_total[label] += 1
            print("Rule: {}".format(rule[0]))
            for i in range(class_len):
                class_correct[i] += local_class_correct[i]
                class_total[i] += local_class_total[i]

                if local_class_total[i] > 0:
                    print('Test Accuracy of %10s: %2d%% (%2d/%2d)' % (
                        classes[i], 100 * local_class_correct[i] / local_class_total[i],
                        np.sum(local_class_correct[i]), np.sum(local_class_total[i])))
                else:
                    print('Test Accuracy of %10s: N/A (no training examples)' % (classes[i]))

            print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
                100. * np.sum(local_class_correct) / np.sum(local_class_total),
                np.sum(local_class_correct), np.sum(local_class_total)))
            print('Loss is {}'.format(rule[-1].item()))
            total_loss += rule[-1].item()
            print("="*20)

    for i in range(class_len):
        if class_total[i] > 0:
            print('Test Accuracy of %10s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %10s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    print('Loss is {}'.format(total_loss))    

class TensorLargeData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        #들어온 x는 tensor형태로 변환
        self.len = self.y_data.shape[0]

    # x,y를 튜플형태로 바깥으로 내보내기
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len