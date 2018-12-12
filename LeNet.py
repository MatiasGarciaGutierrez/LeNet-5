import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class LeNet(nn.Module):
	def __init__(self):

		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, (5,5), padding = 2)

		#In_place is used for changing directly the tensor without having to copy it 
		self.relu1 = nn.ReLU(inplace = True)

		#Creates a layer of max pooling
		self.maxpool1 = nn.MaxPool2d((2,2), stride = (2,2))

		self.conv2 = nn.Conv2d(6, 16, (5,5))

		self.relu2 = nn.ReLU(inplace = True)

		self.maxpool2 = nn.MaxPool2d((2,2), stride = (2,2))

		self.fc1 = nn.Linear(16*5*5, 120)
		self.relu3 = nn.ReLU(inplace = True)

		self.fc2 = nn.Linear(120, 84)
		self.relu4 = nn.ReLU(inplace = True)

		self.fc3 = nn.Linear(84, 10)
		self.softmax = nn.Softmax(dim = 1)

	
	def forward(self, x):

		out = self.conv1(x)
		out = self.relu1(out)
		out = self.maxpool1(out)
		
		out = self.conv2(out)
		out = self.relu2(out)
		out = self.maxpool2(out)

		out = out.view(-1, 16*5*5)

		out = self.fc1(out)
		out = self.relu3(out)

		out = self.fc2(out)
		out = self.relu4(out)

		out = self.fc3(out)

		out = self.softmax(out)

		return out 


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def train_model(model, criterion, optimizer, trainloader, epoch_number = 2):
	finish = False
	for epoch in range(epoch_number):

		running_loss = 0.0
		#enumerate allows for automatic counter in a for loop 
		for i, data in enumerate(trainloader, 0):

			# get the inputs of the data.
			inputs, labels = data

			#Clean the gradients
			optimizer.zero_grad()

			#Forward
			outputs = model(inputs)

			#Calculate loss
			loss = criterion(outputs, labels)

			#Propagates the gradient 
			loss.backward()

			#Updates the weigths of the network
			optimizer.step()

			running_loss += loss.item()

			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
				if (running_loss/2000 < 0.06):
					finish = True
					return

				running_loss = 0.0

	print('Finished Training')
	print("Saving weigths ...")
	torch.save(net.state_dict(), "./backup/2_epochs.pt")
	print("weigths saved!")
	return 
	
if __name__ == '__main__':
	net = LeNet()
	#print(net)
	#input = torch.randn(1, 1, 32, 32)
	#out = net(input)
	#target = torch.randn(10)  
	#print(out)

	#ToTensor transform a PIL image or a numpy array to a torch tensor. Normalize normalizes an image with mean and standard deviations (arrays of channels)
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	#Loads or download a dataset 
	trainset = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform  = transform)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle = True, num_workers = 2)

	testset = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transform)

	testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = False, num_workers = 2)


	classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	train_model(net, criterion, optimizer, trainloader)


	#Mean Squared Error 
	#criterion = nn.MSELoss()
	#loss = criterion(out, target)
	#print(loss.grad_fn)

	#Propagates the gradiant to backward
	#loss.backward()


	#Create an optmizares Stochastic Gradient Descent 
	#optimizer = optim.SGD(net.parameters(), lr = 0.01)









	


