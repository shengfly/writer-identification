#*************************
# Deep Learning Package for Writer Identification
# @author: Sheng He
# @Email: heshengxgd@gmail.com
# Github: https://github.com/shengfly/writer-identification


# Sheng He, Lambert Schomaker,  FragNet: Writer Identification Using Deep Fragment Networks,
# IEEE Transactions on Information Forensics and Security ( Volume: 15), Pages: 3013-3022
# @Arixv: https://arxiv.org/pdf/2003.07212.pdf

#*************************


import torch
import torch.nn as nn

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
        xlist=[x]
        x = self.conv1(x)
        xlist.append(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        xlist.append(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        xlist.append(x)
        x = self.maxp3(x)
        x = self.conv4(x)
        xlist.append(x)
        return xlist


        
class FragNet(nn.Module):
    def __init__(self,inplace,num_classes):
        super().__init__()
        
        self.net = VGGnet(inplace)
        
        layers=[64,128,256,512,512]
        
        self.conv0 = self._conv(inplace,layers[0])
        self.conv1 = self._conv(layers[0]*2,layers[1])
        self.maxp1 = nn.MaxPool2d(2,stride=2)
        self.conv2 = self._conv(layers[1]*2,layers[2])
        self.maxp2 = nn.MaxPool2d(2,stride=2)
        self.conv3 = self._conv(layers[2]*2,layers[3])
        self.maxp3 = nn.MaxPool2d(2,stride=2)
        self.conv4 = self._conv(layers[3]*2,layers[4])
        self.maxp4 = nn.MaxPool2d(2,stride=2)
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512,num_classes)
        
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
    
    def forward(self,x):
        xlist = self.net(x)
        
        step = 16
        
        #print(xlist[0].shape)
        
        # input image
        reslist = []
        for n in range(0,65,step):
            xpatch = xlist[0][:,:,:,n:n+64]
            r = self.conv0(xpatch)
            reslist.append(r)
        
        # 0-layer
        idx = 0
        res1list = []
        for n in range(0,65,step):
            xpatch = xlist[1][:,:,:,n:n+64]
            xpatch = torch.cat([xpatch,reslist[idx]],1)
            idx += 1
            r = self.conv1(xpatch)
            r = self.maxp1(r)
            res1list.append(r)
        
        # 1-layer
        idx = 0
        res2list = []
        step = 8
        for n in range(0,33,step):
            xpatch = xlist[2][:,:,:,n:n+32]
            xpatch = torch.cat([xpatch,res1list[idx]],1)
            idx += 1
            r = self.conv2(xpatch)
            r = self.maxp2(r)
            res2list.append(r)
        
        # 2-layer
        
        idx = 0
        res3list = []
        step = 4
        for n in range(0,17,step):
            xpatch = xlist[3][:,:,:,n:n+16]
            xpatch = torch.cat([xpatch,res2list[idx]],1)
            idx += 1
            r = self.conv3(xpatch)
            r = self.maxp3(r)
            res3list.append(r)
            
        # 3-layer
        idx = 0
        step = 2
        logits_list = []

        for n in range(0,9,step):
            xpatch = xlist[4][:,:,:,n:n+8]
            xpatch = torch.cat([xpatch,res3list[idx]],1)
            idx += 1
            r = self.conv4(xpatch)
            r = self.maxp4(r)
            r = torch.flatten(self.avg(r),1)

            c = self.classifier(r)
            
            logits_list.append(c)
            
        
        combined_logits = 0
        for r in logits_list:
            combined_logits += r
            
        
        return logits_list,combined_logits
        

if __name__ == '__main__':
    x = torch.rand(4,1,64,128)
    
    mod = FragNet(1,105)
    ylist,c = mod(x)
    for y in ylist:
        print(y.shape)
    print(c.shape)
  
    


