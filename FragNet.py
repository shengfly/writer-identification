#*************************
# Deep Learning Package for Writer Identification
# @author: Sheng He
# @Email: heshengxgd@gmail.com
# Github: https://github.com/shengfly/writer-identification
#*************************

'''
@ usage: load the image for PyTorch
'''

import os
import pickle
import numpy as np
import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import torchvision.transforms.functional as TF
from scipy import misc
import random

class WriterDataLoader(data.Dataset):
    def __init__(self,dataset,folder,scale_size=(64,128),is_training=True):
        super().__init__()
        
        self.is_training = is_training
        
        self.scale_size = scale_size
        self.broader = 10
        #self.resize_crop = [i+self.broader for i in self.scale_size]
        
        self.folder = folder
        self.dataset = dataset
        
        self.labelidx_name = self.dataset+'_WRITER_LABEL_INDEX_DotMove.pkl'
        self.imglist = self._get_image_list(self.folder)
        
        self.writer_label = self._convert_identity2index(self.labelidx_name)
        self.num_writer = len(self.writer_label)
        
        print(dataset,' get image file:',len(self.imglist))
        print('get number of writer:',len(self.writer_label))
        
    # convert to idx for neural network
    def _convert_identity2index(self,savename):
        if os.path.exists(savename):
            print('label index dictory existed, load for reuse')
            with open(savename,'rb') as fp:
                identity_idx = pickle.load(fp)
        else:
            identity_idx = {}
            idlist = self._get_all_identity()
            for idx,ids in enumerate(idlist):
                identity_idx[ids] = idx
    
            with open(savename,'wb') as fp:
                pickle.dump(identity_idx,fp)
    
        return identity_idx
            
    def _get_image_list(self,folder,ftype='pickle'):
        flist = os.listdir(folder)
        imglist = []
        for img in flist:
            if img.endswith(ftype):
                imglist.append(img)
        return imglist
    
    
    def _get_all_identity(self):
        writer_list = []
        for img in self.imglist:
            writerId = self._get_identity(img)
            writer_list.append(writerId)
        writer_list=list(set(writer_list))
        return writer_list
    
    def _get_identity(self,fname):
        if self.dataset == 'CERUG':
            return fname.split('_')[0]
        else: return fname.split('-')[0]
    
    def _resize(self,sample,scale_size):
        image = TF.resize(sample,scale_size)
        return image
    
    def transform(self,sample):
        image = Image.fromarray(sample,mode='L')
        
        if self.is_training:
            image = self._resize(image,self.resize_crop)
            top = np.random.randint(self.broader)
            left = np.random.randint(self.broader)
            image = TF.crop(image,top,left,self.scale_size[0],self.scale_size[1])
            if np.random.randint(10) < 5:
                image = TF.hflip(image)
        else:
            image = self._resize(image,self.scale_size)
            
        image = Compose([ToTensor()])(image)
        return image
    
    def _resize_ration(self,image):
        
        h,w = image.shape[:2]
        ratio_h = float(self.scale_size[0])/float(h)
        ratio_w = float(self.scale_size[1])/float(w)
        
        if ratio_h < ratio_w:
            ratio = ratio_h
        else:
            ratio = ratio_w
    
        nh = int(ratio * h)
        nw = int(ratio * w)
        
        imre = misc.imresize(image,(nh,nw))
        
        imre = 255 - imre
        ch,cw = imre.shape[:2]
        
        new_img = np.zeros(self.scale_size)
        
        if self.is_training:
            dy = int((self.scale_size[0]-ch))
            dx = int((self.scale_size[1]-cw))
            dy = random.randint(0,dy)
            dx = random.randint(0,dx)
        else:
            dy = int((self.scale_size[0]-ch)/2.0)
            dx = int((self.scale_size[1]-cw)/2.0)
    
        imre = imre.astype('float')
    
        new_img[dy:dy+ch,dx:dx+cw] = imre
        
        return new_img

    def _loadPickle(self,fname):
        with open(fname,'rb') as fp:
            data = pickle.load(fp)
        return data
    
    def __getitem__(self,index):
                
        imgfile = self.imglist[index]
        writer = self.writer_label[self._get_identity(imgfile)]
        
        image = self._loadPickle(self.folder + imgfile)
        #print('image:',image.shape)
        image = self._resize_ration(image)
        #image = self.transform(image)
        
        image = Compose([ToTensor()])(image)
        #print(image.max())
        #image = 1.0 - image
        writer = torch.from_numpy(np.array(writer))

        return image,writer
            
    def __len__(self):
        return len(self.imglist)


if __name__ == '__main__':
    loader = WriterDataLoader('IAM','../residualReLU/IAM_pickle/test/',is_training=True)
    import matplotlib.pyplot as plt
    for n in range(0,len(loader)):
        image,writer = loader.__getitem__(n)
        image = image[0].cpu().numpy()
        plt.imshow(image)
        print(image.shape,writer)
        break

