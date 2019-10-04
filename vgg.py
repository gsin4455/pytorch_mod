'''VGG10 in Pytorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHAN = 1024



class vgg(nn.Module):
    def __init__(self, filts):
        super(vgg, self).__init__()
        self.features = self._make_layers(filts[0:7],IN_CHAN)
        self.dense  = self._make_denselayers(filts[7:9], 576)
        self.fc = nn.Linear(filts[8],filts[9])

    def forward(self, x):
        out = self.features(x)
        #flatten
        out = out.view(out.size(0), -1)
        #dense layers
        out = self.dense(out)
        #return out
        #final activation layer
        out = self.fc(out)
        
        return out
    

    def _make_denselayers(self,filts,in_channels):
         
        layers = []
        
        #dense layers
        for x in filts:
            layers += [nn.Linear(in_channels, x), nn.BatchNorm1d(x),nn.ReLU(inplace=True)]
            in_channels = x
        return nn.Sequential(*layers)

    def _make_layers(self, filts,in_chan):
        layers = []
        in_channels = in_chan
        i = 0
        for x in filts:
            i +=1
            layers += [nn.Conv1d(in_channels, x, kernel_size=3,padding=1),
                nn.MaxPool1d(kernel_size=2, stride=1,padding=1), 
                nn.BatchNorm1d(x),
                nn.ReLU(inplace=True)]
            in_channels = x
            
        return nn.Sequential(*layers)




if __name__=='__main__':
    #Test code
    filts =  [64]*7 + [512, 512, 24]
    net = vgg(filts)
    x = torch.randn(128,2,128)
    y = net(x)
    print(y)
    
