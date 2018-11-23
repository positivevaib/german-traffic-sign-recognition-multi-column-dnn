# import dependencies
import argparse
import numpy as np
import os
import PIL
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import data
import dnn

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type = str, default = os.getcwd(), help = 'absolute path to datasets')
parser.add_argument('-l', '--load', action = 'store_true', help = 'load pre-trained model parameters')
parser.add_argument('-m', '--model', type = str, default = os.getcwd(), help = 'absolute path to model parameters')
parser.add_argument('-p', '--preprocess', action = 'store_true', help = 'preprocess data')

args = parser.parse_args()

# preprocess data
if args.preprocess:
	print('preprocessing\n')
	data.preprocess_images(args.data)
	data.split_dataset(os.path.join(args.data, 'training_set'))
	print()

# load training data
print('loading training data')
training_set = data.load_dataset(args.data, dataset = 'training', training_batch_size = 32)
original_training_loader = training_set['original']
imadjust_training_loader = training_set['imadjust']
histeq_training_loader = training_set['histeq']
adapthisteq_training_loader = training_set['adapthisteq']

# load validation data
print('loading validation data')
validation_set = data.load_dataset(args.data, dataset = 'validation')
original_validation_loader = validation_set['original']
imadjust_validation_loader = validation_set['imadjust']
histeq_validation_loader = validation_set['histeq']
adapthisteq_validation_loader = validation_set['adapthisteq']

# load test data
print('loading test data\n')
test_set = data.load_dataset(args.data, dataset = 'test')
original_test_loader = test_set['original']
imadjust_test_loader = test_set['imadjust']
histeq_test_loader = test_set['histeq']
adapthisteq_test_loader = test_set['adapthisteq']

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, '\n')

# train deep neural nets
if args.load:
	print('loading pre-trained model parameters\n')
else:
	print('training deep neural nets')

for net_name in nets.keys():
	net = nets[net_name]

	if args.load:
		print(net_name)
		net.load_state_dict(torch.load(os.path.join(args.model, (net_name + '.pth')), map_location = device))
	else:
		net.apply(dnn.parameters_init)
		net.to(device)

		print('\n' + net_name)

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
		loss_file.write('epoch,training_loss,validation_loss\n')

		# train deep neural net
		validation_loss_prev = None

		total_epochs = 15
		for epoch in range(total_epochs):
			print('epoch', epoch + 1, 'of', total_epochs)

			batch_idx = None
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

				# initial training and validation loss
				if epoch == 0 and batch_idx == 0:
					net.to('cpu')
					with torch.no_grad():
						validation_inputs, validation_labels = next(iter(validation_loader))
						validation_loss_prev = criterion(net(validation_inputs), validation_labels)
					net.to(device)

					loss_file.write(str(epoch) + ',' + str(running_loss) + ',' + str(validation_loss_prev.item()) + '\n')

					print('initial training loss:', running_loss)
					print('initial validation loss:', validation_loss_prev.item())

			# save training and validation loss to file
			net.to('cpu')
			with torch.no_grad():
				validation_inputs, validation_labels = next(iter(validation_loader))
				validation_loss = criterion(net(validation_inputs), validation_labels)
			net.to(device)

			loss_file.write(str(epoch + 1) + ',' + str(running_loss/batch_idx + 1) + ',' + str(validation_loss.item()) + '\n')

			# print current loss
			print('training loss:', running_loss/(batch_idx + 1))
			print('validation loss:', validation_loss.item())

			# apply early stopping
			if validation_loss > validation_loss_prev:
				print('early stopping applied')
				break
			
			validation_loss_prev = validation_loss

			# save trained parameters
			torch.save(net.state_dict(), os.path.join(args.model, (net_name + '.pth')))

# evaluate deep neural nets and create multi-column deep neural net
print('\nevaluating deep neural nets and creating multi-column deep neural net\n')

# create csv file to track accuracy
eval_file = open('eval.csv', 'w+')
eval_file.write('net,accuracy\n')

with torch.no_grad():
	mcdnn_outputs = torch.zeros((next(iter(original_test_loader))[0].size()[0], 43))

for net_name in nets.keys():
	net = nets[net_name]

	if 'original' in net_name:
		test_loader = original_test_loader
	elif 'imadjust' in net_name:
		test_loader = imadjust_test_loader
	elif 'histeq' in net_name:
		test_loader = histeq_test_loader
	else:
		test_loader = adapthisteq_test_loader

	# compute neural net accuracy and add outputs to mcdnn_outputs
	net.to('cpu')
	with torch.no_grad():
		test_inputs, test_labels = next(iter(test_loader))
		test_labels.tolist()

		outputs = net(test_inputs)

		mcdnn_outputs.add_(outputs)

		_, predictions = torch.max(outputs, dim = 1)
		predictions.tolist()

		accuracy = 0
		for i in range(len(test_labels)):
			if predictions[i] == test_labels[i]:
				accuracy += 1
		accuracy = accuracy/len(test_labels)
	net.to(device)

	# print neural net accuracy
	print(net_name, 'accuracy:', round(accuracy*100, 2))

	# save neural net accuracy to file
	eval_file.write(net_name + ',' + str(round(accuracy*100, 2)) + '\n')

# compute mcdnn predictions
with torch.no_grad():
	mcdnn_outputs.div_(len(nets))
	_, mcdnn_predictions = torch.max(mcdnn_outputs, dim = 1)
	mcdnn_predictions.tolist()

# evaluate multi-column deep neural net
_, test_labels = next(iter(original_test_loader))
test_labels.tolist()

accuracy = 0
for i in range(len(test_labels)):
	if mcdnn_predictions[i] == test_labels[i]:
		accuracy += 1
accuracy = accuracy/len(test_labels)

# print mcdnn accuracy
print('\nmcdnn accuracy:', round(accuracy*100, 2))

# save mcdnn accuracy to file
eval_file.write('mcdnn,' + str(round(accuracy*100, 2)))

# save predictions
file_out = open('mcdnn_predictions.csv', 'w+')
np.savetxt(file_out, mcdnn_predictions.numpy(), delimiter = ',')