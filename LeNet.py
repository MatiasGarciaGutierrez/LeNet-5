import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
	def __init__(self):

		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, (5,5))

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

if __name__ == '__main__':
	net = LeNet()
	print(net)
	input = torch.randn(1, 1, 32, 32)
	out = net(input)
	print(out)


