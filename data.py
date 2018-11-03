# import dependencies
import torch
import torchvision
import os

# load training data
training_data_original = torchvision.datasets.ImageFolder(root = 'Final_Training/Images', transform = torchvision.transforms.ToTensor())
training_data_imadjust = torchvision.datasets.ImageFolder(root = 'Final_Training/imadjust', transform = torchvision.transforms.ToTensor())
training_data_histeq = torchvision.datasets.ImageFolder(root = 'Final_Training/histeq', transform = torchvision.transforms.ToTensor())
training_data_adapthisteq = torchvision.datasets.ImageFolder(root = 'Final_Training/adapthisteq', transform = torchvision.transforms.ToTensor())

# load test data
