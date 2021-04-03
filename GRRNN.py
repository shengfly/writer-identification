"""

    This code is for the following paper:
    
    Sheng He and Lambert Schomaker
    SGR-RNN: Global-Context Residual Recurrent Neural Networks for Writer Identification
    Pattern Recognition
    
    @email: heshengxgd@gmail.com
    @author: Sheng He
    @Github: https://github.com/shengfly/writer-identification
    
"""
  
import torch
import torch.nn as nn
import math

class VGGnet(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
        layers=[64,128,256,512]
        self.conv1 = self._conv(input_channel,layers[0])
        self.maxp1 = nn.MaxPool2d(2,stride=2)
        self.conv2 = self._conv(layers[0],layers[1])
        self.maxp2 = nn.MaxPool2d(2,stride=2)
        self.conv3 = self._conv(layers[1],layers[2])
        self.maxp3 = nn.MaxPool2d(2,stride=2)
        self.conv4 = self._conv(layers[2],layers[3])
        self.maxp4 = nn.MaxPool2d(2,stride=2)
        
        self.featv = nn.AdaptiveAvgPool2d((1,2))
        self.avg = nn.AdaptiveAvgPool2d(1)
        
    def _conv(self,inplance,outplance,nlayers=2):
        conv = []
        for n in range(nlayers):
            conv.append(nn.Conv2d(inplance,outplance,kernel_size=3,
                          stride=1,padding=1,bias=False))
            conv.append(nn.BatchNorm2d(outplance))
            conv.append(nn.ReLU(inplace=True))
            inplance = outplance
            
        conv = nn.Sequential(*conv)
               
        return conv
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.maxp3(x)
        xfeat = x
        x = self.conv4(x)
        x = self.maxp4(x)
        x = torch.flatten(self.avg(x),1)
        return x,xfeat


        
class GRUcell(nn.Module):
    def __init__(self,inplance,hidden_size,bias=True):
        super().__init__()
        
        self.inplance = inplance
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.x2h = nn.Linear(inplance,3*hidden_size,bias=bias)
        self.h2h = nn.Linear(hidden_size,3*hidden_size,bias=bias)
        
        self.reset_parameters()
        
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std,std)
    
    def forward(self,x,hidden):
        x = x.view(-1,x.size(1))
        
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        
        
        i_r,i_i,i_n = gate_x.chunk(3,1)
        h_r,h_i,h_n = gate_h.chunk(3,1)
        
        resetgate = torch.sigmoid(i_r+h_r)
        inputgate = torch.sigmoid(i_i+h_i)
        
        newgate = torch.tanh(i_n + (resetgate*h_n))
        hy = (1-inputgate) * newgate + inputgate * hidden
        
        return hy

        
class GrnnNet(nn.Module):
    def __init__(self,input_channel,num_classes=105,mode='global'):
        super().__init__()
        
        self.mode = mode
    
        self.net = VGGnet(input_channel)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.ada = nn.Linear(256,512)
        
        self.rnn = GRUcell(512,512)
        self.classifier = nn.Linear(512,num_classes)
       
    def forward(self,x):
        glf,feat = self.net(x)

        if 'vertical' in self.mode:
            seq = feat.size()[-1]//2
            
            for n in range(seq):
                s=2*n
                patch = feat[:,:,:,s:s+2]
                lx = torch.flatten(self.avg(patch),1)
                lx = self.ada(lx)

                glf =  self.rnn(lx,glf) + lx

                if n == 0:
                    glfa = glf 
                else:
                    glfa = glfa + glf
            
            logits = self.classifier(glfa)

        if 'horzontal' in self.mode:
            seq = feat.size()[-2]
            
            for n in range(seq):
                patch = feat[:,:,n,:].unsqueeze(2)
                lx = torch.flatten(self.avg(patch),1)
                lx = self.ada(lx)
                glf =  self.rnn(lx,glf)+lx

                if n == 0:
                    glfa = glf 
                else:
                    glfa = glfa + glf
            
            logits = self.classifier(glfa)
        
        return logits

    

if __name__ == '__main__':
    
    x = torch.rand(1,1,64,128)
    
    mod = GrnnNet(1,105,mode='vertical')
    
    logits = mod(x)
    
    print(logits.shape)
    
    
    
