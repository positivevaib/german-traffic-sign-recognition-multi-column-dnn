# import dependencies
import os
import pandas as pd
from PIL import Image
import matlab.engine
import torch
import torchvision

def preprocess_data():
    '''crop images and call matlab preprocessing script'''
    print('resizing images')

    for image_class in os.listdir('training_set/original'):
        if not image_class.startswith('.'):
            annotations = pd.read_csv('training_set/original/' + image_class + '/GT-' + image_class + '.csv', sep = ';')
            annotations = annotations.set_index('Filename')
            for file in os.listdir('training_set/original/' + image_class):
                if file.startswith('000'):
                    x1 = annotations.at[file, 'Roi.X1']
                    y1 = annotations.at[file, 'Roi.Y1']
                    x2 = annotations.at[file, 'Roi.X2']
                    y2 = annotations.at[file, 'Roi.Y2']

                    image = Image.open('training_set/original/' + image_class + '/' + file)
                    image = image.crop((x1, y1, x2, y2))
                    image = image.resize((48, 48), resample = Image.BILINEAR)

    print('preprocessing')

    eng = matlab.engine.start_matlab()
    eng.preprocessing(nargout = 0)
    eng.exit()

def load_data():
    '''load data'''
    # create validation split
    print('creating validation set')

    os.mkdir('validation_set')
    for file in os.listdir('training_set'):
        if not file.startswith('.'):
            os.mkdir('validation_set/' + file)
            for image_class in os.listdir('training_set/' + file):
                if image_class.startswith('000'):
                    os.mkdir('validation_set/' + file + '/' + image_class)
                    for image in os.listdir('training_set/' + file + '/' + image_class):
                        if image.startswith('00000') or image.startswith('00001') or image.startswith('00002'):
                            os.rename('training_set/' + file + '/' + image_class + '/' + image, 'validation_set/' + file + '/' + image_class + '/' + image)

    print('loading training data')

    # load training data
    training_data = {}
    for file in os.listdir('training_set'):
        if not file.startswith('.'):
            training_data[file] = torchvision.datasets.ImageFolder(root = 'training_set/' + file, transform = torchvision.transforms.ToTensor())

    print('loading validation data')

    # load validation data
    validation_data = {}
    for file in os.listdir('validation_set'):
        if not file.startswith('.'):
            validation_data[file] = torchvision.datasets.ImageFolder(root = 'validation_set/' + file, transform = torchvision.transforms.ToTensor())

    # return data
    return [training_data, validation_data]