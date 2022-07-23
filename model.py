
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size,512)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(512, 256)
        self.l3=nn.Linear(256,128)
        self.l4=nn.Linear(128,num_classes)
        self.bn1=nn.BatchNorm1d(512)
        self.bn2=nn.BatchNorm1d(256)
        
    def forward(self,x):
        out=self.l1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.l2(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.l3(out)
        out=self.relu(out)
        out=self.l4(out)
        out=nn.functional.log_softmax(out,dim=1)
        return out