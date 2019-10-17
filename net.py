'''simple cnn in Pytorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHAN = 2048



class net(nn.Module):
    def __init__(self, classes):
        super(net, self).__init__()
        self.conv1 = nn.Conv1d(IN_CHAN,64,kernel_size=2,stride=2,padding=2)
        self.conv2 = nn.Conv1d(64,96,kernel_size=3,stride=2,padding=2)
        self.pool = nn.MaxPool1d(3,stride=2)

        self.conv3 = nn.Conv1d(96,128, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=2)
        
        self.fc = nn.Linear(1280, 48)
        self.do = nn.Dropout()
        self.out = nn.Linear(48, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc(self.do(x)))
    
        x = self.out(x)
        return x

if __name__=='__main__':
    #Test code
    nn = net(2)
    x = torch.randn(128,1024,2)
    y = nn(x)
    print(y)
    
