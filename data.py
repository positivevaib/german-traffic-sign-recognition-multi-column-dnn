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
    '''crop images according to annotated bounding boxes and resize'''
    annotations = pd.read_csv(glob.glob(os.path.join(path, '*.csv'))[0],
                              sep=';')
    annotations = annotations.set_index('Filename')

    for image_name in os.listdir(path):
        if image_name.endswith('.ppm'):
            x1 = annotations.at[image_name, 'Roi.X1']
            y1 = annotations.at[image_name, 'Roi.Y1']
            x2 = annotations.at[image_name, 'Roi.X2']
            y2 = annotations.at[image_name, 'Roi.Y2']

            image = Image.open(os.path.join(path, image_name))
            image = image.crop((x1, y1, x2, y2))
            image = image.resize((48, 48), resample=Image.BILINEAR)
            image.save(os.path.join(path, image_name))


def normalize_images(path):
    '''run matlab script to normalize images'''
    eng = matlab.engine.start_matlab()
    eng.normalize(path, nargout=0)
    eng.exit()


def preprocess_images(path):
    '''preprocess images'''
    for image_class in os.listdir(
            os.path.join(path, 'training_set', 'original')):
        if os.path.isdir(
                os.path.join(path, 'training_set', 'original', image_class)):
            crop_and_resize_images(
                os.path.join(path, 'training_set', 'original', image_class))
    normalize_images(os.path.join(path, 'training_set'))


def split_dataset(path, tracks=3):
    '''split dataset into training, validation and test sets'''
    os.mkdir(os.path.join(os.path.dirname(path), 'validation_set'))
    os.mkdir(os.path.join(os.path.dirname(path), 'test_set'))
    for file_name in os.listdir(path):
        if os.path.isdir(os.path.join(path, file_name)):
            os.mkdir(
                os.path.join(os.path.dirname(path), 'validation_set',
                             file_name))
            os.mkdir(os.path.join(os.path.dirname(path), 'test_set',
                                  file_name))
            for image_class in os.listdir(os.path.join(path, file_name)):
                if os.path.isdir(os.path.join(path, file_name, image_class)):
                    os.mkdir(
                        os.path.join(os.path.dirname(path), 'validation_set',
                                     file_name, image_class))
                    os.mkdir(
                        os.path.join(os.path.dirname(path), 'test_set',
                                     file_name, image_class))
                    for image in os.listdir(
                            os.path.join(path, file_name, image_class)):
                        if image.endswith('.ppm'):
                            for idx in range(tracks):
                                if '0' + str(idx) + '_' in image:
                                    if idx != tracks - 1:
                                        os.rename(
                                            os.path.join(
                                                path, file_name, image_class,
                                                image),
                                            os.path.join(
                                                os.path.dirname(path),
                                                'validation_set', file_name,
                                                image_class, image))
                                    else:
                                        os.rename(
                                            os.path.join(
                                                path, file_name, image_class,
                                                image),
                                            os.path.join(
                                                os.path.dirname(path),
                                                'test_set', file_name,
                                                image_class, image))


def load_dataset(path, dataset, training_batch_size=1):
    '''load dataset'''
    if dataset == 'training':
        data = {}
        for file_name in os.listdir(os.path.join(path, 'training_set')):
            if os.path.isdir(os.path.join(path, 'training_set', file_name)):
                data[file_name] = torch.utils.data.DataLoader(
                    torchvision.datasets.ImageFolder(
                        root=os.path.join(path, 'training_set', file_name),
                        transform=torchvision.transforms.ToTensor()),
                    batch_size=training_batch_size,
                    shuffle=True)
        return data

    elif dataset == 'validation':
        data = {}
        for file_name in os.listdir(os.path.join(path, 'validation_set')):
            if os.path.isdir(os.path.join(path, 'validation_set', file_name)):
                validation_set = torchvision.datasets.ImageFolder(
                    root=os.path.join(path, 'validation_set', file_name),
                    transform=torchvision.transforms.ToTensor())
                data[file_name] = torch.utils.data.DataLoader(
                    validation_set, batch_size=len(validation_set))
        return data

    elif dataset == 'test':
        data = {}
        for file_name in os.listdir(os.path.join(path, 'test_set')):
            if os.path.isdir(os.path.join(path, 'test_set', file_name)):
                test_set = torchvision.datasets.ImageFolder(
                    root=os.path.join(path, 'test_set', file_name),
                    transform=torchvision.transforms.ToTensor())
                data[file_name] = torch.utils.data.DataLoader(
                    test_set, batch_size=len(test_set))
        return data
