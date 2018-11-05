# import dependencies
from data import preprocess_data, load_data
from dnn import Net1, Net2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import random


# preprocess and load data
preprocess_data()

data = load_data()
training_set = data[0]
validation_set = data[1]
test_set = data[2]

training_loader = torch.utils.data.DataLoader(training_set, batch_size = 35, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_set)

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

# train DNNs
print('training DNNs')
for net_name in nets.values():
	net = nets[net_name]

    # define loss function and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters())
    
	print('training', net_name)
	for epoch in range(30):
		running_loss = 0
		for batch, data in enumerate(training_loader, 0):
			inputs, labels = data
			for i in range(len(inputs)):
				perturbations = [torchvision.transforms.RandomRotation(degrees = 5, resample = PIL.Image.BILINEAR), torchvision.transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1), resample = PIL.Image.BILINEAR), torchvision.transforms.RandomResizedCrop(size = 48, ratio = (1, 1))]
				inputs[i] = random.choice(perturbations)(inputs[i])

			optimizer.zero_grad()

			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			if (batch % 2000 == 1999):
				print('[', epoch + 1 + ', ', batch + 1, '] loss: ', running_loss/2000, sep = '')

# define MCDNN