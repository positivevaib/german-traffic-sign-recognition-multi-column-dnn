# import dependencies
import glob
import os

import matlab.engine
import pandas as pd
import PIL.Image as Image
import torch
import torchvision

# define functions
def crop_and_resize_images(path):
    '''crop images according to annotated bounding boxes and resize to given dimensions'''
    annotations = pd.read_csv(glob.glob(os.path.join(path, '*.csv')))
    annotations = annotations.set_index('Filename')

    for image_name in glob.glob(path, '*.ppm'):
        x1 = annotations.at[image_name, 'Roi.X1']
        y1 = annotations.at[image_name, 'Roi.Y1']
        x2 = annotations.at[image_name, 'Roi.X2']
        y2 = annotations.at[image_name, 'Roi.Y2']

        image = Image.open(os.path.join(path, image_name))
        image = image.crop((x1, y1, x2, y2))
        image = image.resize((48, 48), resample = Image.BILINEAR)
        image.save(os.path.join(path, image_name))

def normalize_images(path):
    '''run matlab script to normalize images'''
    eng = matlab.engine.start_matlab()
    eng.preprocessing(path, nargout = 0)
    eng.exit()

def preprocess_images(path, test_set = False):
    '''preprocess images'''
    if not test_set:
        for image_class in os.listdir(os.path.join(path, os.listdir(path)[0])):
            crop_and_resize_images(os.path.join(path, os.listdir(path)[0], image_class))
            normalize_images(path)
    else:
        crop_and_resize_images(os.path.join(path, os.listdir(path)[0]))

def split_dataset(path, prefixes = 3):
    '''split dataset into training and validation sets'''
    if not os.path.isdir(os.path.join(os.path.dirname(path), 'validation_set')):
        os.mkdir(os.path.join(os.path.dirname(path), 'validation_set'))
        for file in os.listdir(path):
            if os.path.isdir(file):
                os.mkdir(os.path.join(os.path.dirname(path), 'validation_set', file))
                for image_class in os.listdir(os.path.join(path, file)):
                    if os.path.isdir(os.path.join(path, file, image_class)):
                        os.mkdir(os.path.join(os.path.dirname(path), 'validation_set', file, image_class))
                        for image in os.listdir(os.path.join(path, file, image_class)):
                            for idx in range(prefixes):
                                if image.startswith('0000' + str(idx)):
                                    os.rename(os.path.join(path, file, image_class, image), os.path.join(os.path.dirname(path), file, image_class, image))

def load_dataset(path, dataset, training_batch_size = 1):
    '''load dataset'''
    if dataset = 'training':
        data = {}
        for file in os.listdir(os.path.join(path, 'training_set')):
            if os.path.isdir(os.path.join(path, 'training_set', file)):
                data[file] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root = os.path.join(path, 'test_set', file)), transform = torchvision.transforms.ToTensor(), batch_size = training_batch_size)
    elif dataset == 'validation':
        data = {}
        for file in os.listdir(os.path.join(path, 'validation_set')):
            if os.path.isdir(os.path.join(path, 'validation_set', file)):
                validation_set = torchvison.datasets.ImageFolder(root = os.path.join(path, 'validation_set', file), transform = torchvision.transforms.ToTensor())
                data[file] = torch.utils.data.DataLoader(validation_set, batch_size = len(validation_set))
    else:
        data = []
        for file in os.listdir(os.path.join(path, 'test_set')):
            if os.path.isdir(os.path.join(path, 'test_set', file)):
                for image_name in glob.glob(os.path.join(path, 'test_set', file), '*.ppm'):
                    image = Image.open(os.path.join(path, 'test_set', file, image_name))
                    image = torchvision.transforms.ToTensor()(image)
                    data.append(image)
        data = torch.utils.data.DataLoader(data, batch_size = len(data))

    return data