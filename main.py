# import dependencies
import argparse
import numpy as np
import os
import PIL
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import data
import dnn

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type = str, default = os.getcwd(), help = 'absolute path to datasets')
parser.add_argument('-p', '--preprocess', action = 'store_true', help = 'preprocess data')

args = parser.parse_args()

# current path
path = args.data

# preprocess data
if args.preprocess:
	print('preprocessing')
	#data.preprocess_images(path, dataset = 'training')
	#data.preprocess_images(path, dataset = 'test')

	data.split_dataset(os.path.join(path, 'training_set'))

# load training data
print('\rloading training data')
training_set = data.load_dataset(path, dataset = 'training', training_batch_size = 32)
original_training_loader = training_set['original']
imadjust_training_loader = training_set['imadjust']
histeq_training_loader = training_set['histeq']
adapthisteq_training_loader = training_set['adapthisteq']

# load validation data
print('\rloading validation data')
validation_set = data.load_dataset(path, dataset = 'validation')
original_validation_loader = validation_set['original']
imadjust_validation_loader = validation_set['imadjust']
histeq_validation_loader = validation_set['histeq']
adapthisteq_validation_loader = validation_set['adapthisteq']

# load test data
print('\rloading test data')
test_loader = data.load_dataset(path, dataset = 'test')

# instantiate deep neural nets
nets = {}
nets['original_net1'] = dnn.Net1()
nets['original_net2'] = dnn.Net2()

nets['imadjust_net1'] = dnn.Net1()
nets['imadjust_net2'] = dnn.Net2()

nets['histeq_net1'] = dnn.Net1()
nets['histeq_net2'] = dnn.Net2()

nets['adapthisteq_net1'] = dnn.Net1()
nets['adapthisteq_net2'] = dnn.Net2()

# setup device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# train deep neural nets
print('\rtraining deep neural nets')
for net_name in nets.keys():
	net = nets[net_name]
	net.apply(dnn.parameters_init)
	net.to(device)

	print('\r', net_name)

	if 'original' in net_name:
		training_loader = original_training_loader
		validation_loader = original_validation_loader
	elif 'imadjust' in net_name:
		training_loader = imadjust_training_loader
		validation_loader = imadjust_validation_loader
	elif 'histeq' in net_name:
		training_loader = histeq_training_loader
		validation_loader = histeq_validation_loader
	else:
		training_loader = adapthisteq_training_loader
		validation_loader = adapthisteq_validation_loader

    # define loss function and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters())
    
	# create csv file to track training and validation loss
	loss_file = open(net_name + '_' + 'loss.csv', 'w+')
	loss_file.write('epoch,batch,training_loss\n')

	# train deep neural net
	for epoch in range(30):
		print('epoch:', epoch + 1)

		running_loss = 0
		for batch_idx, data in enumerate(training_loader):
			inputs = data[0]
			labels = data[1]

			# apply perturbations
			for i in range(len(inputs)):
				perturbation = random.choice([0, 1, 2, 3])
				if perturbation == 1:
					image = transforms.ToPILImage()(inputs[i])
					image = transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1), resample = PIL.Image.BILINEAR)(image)
					inputs[i] = transforms.ToTensor()(image)
				elif perturbation == 2:
					image = transforms.ToPILImage()(inputs[i])
					image = transforms.RandomResizedCrop(size = 48, ratio = (1, 1))(image)
					inputs[i] = transforms.ToTensor()(image)
				elif perturbation == 3:
					image = transforms.ToPILImage()(inputs[i])
					image = transforms.RandomRotation(degrees = 5, resample = PIL.Image.BILINEAR)(image)
					inputs[i] = transforms.ToTensor()(image)
				else:
					continue

			inputs = inputs.to(device)
			labels = labels.to(device)

			# optimize
			optimizer.zero_grad()

			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			# save training and validation loss to file
			if batch_idx % 10 == 9:
				loss_file.write(str(epoch + 1) + ',' + str(batch_idx + 1) + ',' + str(running_loss/10) + '\n')
				running_loss = 0

		# print current loss
		print('training loss:', loss.item())

		# apply early stopping
		net.to('cpu')
		validation_inputs, validation_labels = next(iter(validation_loader))
		validation_loss = criterion(net(validation_inputs), validation_labels)
		net.to(device)

		print('validation loss:', validation_loss.item())

		if (validation_loss - 0) < sys.float_info.epsilon:
			break

	# save trained parameters
	torch.save(net.state_dict(), path)

# create multi-column deep neural net
dnn_outputs = []
for net in nets.values():
	dnn_outputs.append(net(test_loader))

# average results
mcdnn_output = dnn_outputs[0]
for dnn_output in dnn_outputs[1:]:
	mcdnn_output.add_(dnn_output)

mcdnn_output.div_(len(dnn_outputs))

# compute predictions
_, predictions = torch.argmax(mcdnn_output, dim = 0)

# save predictions
file_out = open('mcdnn_predictions.csv', 'w+')
np.savetxt(file_out, predictions.numpy(), delimiter = ',')