# import dependencies
import os
import pandas as pd
from PIL import Image
import matlab.engine
import torch
import torchvision

# define preprocessing and data loading functions
def preprocess_data():
    '''crop images and call matlab preprocessing script'''
    # crop and resize images
    print('resizing images')
    # training set
    for image_class in os.listdir('training_set/original'):
        if not image_class.startswith('.'):
            annotations = pd.read_csv('training_set/original/' + image_class + '/GT-' + image_class + '.csv', sep = ';')
            annotations = annotations.set_index('Filename')

            for image_name in os.listdir('training_set/original/' + image_class):
                if image_name.endswith('.ppm'):
                    x1 = annotations.at[image_name, 'Roi.X1']
                    y1 = annotations.at[image_name, 'Roi.Y1']
                    x2 = annotations.at[image_name, 'Roi.X2']
                    y2 = annotations.at[image_name, 'Roi.Y2']

                    image = Image.open('training_set/original/' + image_class + '/' + image_name)
                    image = image.crop((x1, y1, x2, y2))
                    image = image.resize((48, 48), resample = Image.BILINEAR)
                    image.save('training_set/original/' + image_class + '/' + image_name)

    # test set
    annotations = pd.read_csv('test_set/images/GT-final_test.test.csv', sep = ';')
    annotations = annotations.set_index('Filename')

    for image_name in os.listdir('test_set/images'):
        if image_name.endswith('.ppm'):
            x1 = annotations.at[image_name, 'Roi.X1']
            y1 = annotations.at[image_name, 'Roi.Y1']
            x2 = annotations.at[image_name, 'Roi.X2']
            y2 = annotations.at[image_name, 'Roi.Y2']

            image = Image.open('test_set/images/' + image_name)
            image = image.crop((x1, y1, x2, y2))
            image = image.resize((48, 48), resample = Image.BILINEAR)
            image.save('test_set/images/' + image_name)

    # normalize images
    print('normalizing images')
    eng = matlab.engine.start_matlab()
    eng.preprocessing(nargout = 0)
    eng.exit()

def load_training_validation_data():
    '''load training and validation data'''
    # create validation split
    if not os.path.isdir('validation_set'):
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

    # load training data
    print('loading training data')
    training_data = {}
    for file in os.listdir('training_set'):
        if not file.startswith('.'):
            training_data[file] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root = 'training_set/' + file, transform = torchvision.transforms.ToTensor()), batch_size = 16, shuffle = True)

    # load validation data
    print('loading validation data')
    validation_data = {}
    for file in os.listdir('validation_set'):
        if not file.startswith('.'):
            validation_set = torchvision.datasets.ImageFolder(root = 'validation_set/' + file, transform = torchvision.transforms.ToTensor())
            validation_data[file] = torch.utils.data.DataLoader(validation_set, batch_size = len(validation_set))

    # return data
    return [training_data, validation_data]

def load_test_data():
    '''load test data'''
    print('loading test data')
    test_data = []
    for image_name in os.listdir('test_set/images'):
        if image_name.endswith('.ppm'):
            image = Image.open('test_set/images/' + image_name)
            image = torchvision.transforms.ToTensor()(image)
            test_data.append(image)

    test_data = torch.utils.data.DataLoader(test_data, batch_size = len(test_data))
    
    # return data
    return test_data