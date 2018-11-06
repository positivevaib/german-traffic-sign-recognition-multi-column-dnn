# import dependencies
import os
from data import preprocess_data, load_training_validation_data, load_test_data
from dnn import Net1, Net2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import PIL
import random

# preprocess data
if not os.path.isdir('validation_set'):
	preprocess_data()

# load data
data = load_training_validation_data()

# training data
training_set = data[0]
original_training_loader = training_set['original']
imadjust_training_loader = training_set['imadjust']
histeq_training_loader = training_set['histeq']
adapthisteq_training_loader = training_set['adapthisteq']

# validation data
validation_set = data[1]
original_validation_loader = validation_set['original']
imadjust_validation_loader = validation_set['imadjust']
histeq_validation_loader = validation_set['histeq']
adapthisteq_validation_loader = validation_set['adapthisteq']

# test data
test_set = load_test_data()
test_loader = torch.utils.data.DataLoader(test_set, batch_size = len(test_set))

# instantiate DNNs
nets = {}
nets['original_net1'] = Net1()
nets['original_net2'] = Net2()

nets['imadjust_net1'] = Net1()
nets['imadjust_net2'] = Net2()

nets['histeq_net1'] = Net1()
nets['histeq_net2'] = Net2()

nets['adapthisteq_net1'] = Net1()
nets['adapthisteq_net2'] = Net2()

# setup device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# train DNNs
print('training DNNs')
for net_name in nets.keys():
	net = nets[net_name]
	net.to(device)

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
    
	# train net
	print('training', net_name)
	
	for epoch in range(50):
		for batch_id, data in enumerate(training_loader):
			inputs = data[0]
			labels = data[1]

			# apply perturbations
			for i in range(len(inputs)):
				perturbation = random.choice([0, 1, 2, 3])
				if perturbation == 1:
					image = torchvision.transforms.ToPILImage()(inputs[i])
					image = torchvision.transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1), resample = PIL.Image.BILINEAR)(image)
					inputs[i] = torchvision.transforms.ToTensor()(image)
				elif perturbation == 2:
					image = torchvision.transforms.ToPILImage()(inputs[i])
					image = torchvision.transforms.RandomResizedCrop(size = 48, ratio = (1, 1))(image)
					inputs[i] = torchvision.transforms.ToTensor()(image)
				elif perturbation == 3:
					image = torchvision.transforms.ToPILImage()(inputs[i])
					image = torchvision.transforms.RandomRotation(degrees = 5, resample = PIL.Image.BILINEAR)(image)
					inputs[i] = torchvision.transforms.ToTensor()(image)
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
		
		# print current loss
		print('epoch:', epoch + 1)
		print('training loss:', loss.item())

		# check validation loss
		validation_inputs, validation_labels = next(iter(validation_loader))
		net.to('cpu')
		validation_loss = criterion(net(validation_inputs), validation_labels)
		net.to(device)

		print('validation loss:', validation_loss.item())

		if validation_loss == 0:
			break

# define MCDNN
