
from os import listdir
import os
from os.path import join
from tkinter import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose,RandomCrop,ToTensor,ToPILImage,CenterCrop,Resize

from PIL import Image


def is_img(x):
    return any(x.endswith(extension) for extension in ['.png','.jpeg','.jpg','.PNG','.JPEG','.JPEG'])

def valid_crop_size(cropsize,scale):
    return cropsize-(cropsize%scale)

def hr_transform(cropsize):
    return Compose([
        RandomCrop(cropsize),
        ToTensor()
    ])

def lr_transform(cropsize,scaling):
    return Compose([
        ToPILImage(),
        Resize(cropsize//scaling, interpolation=Image.BICUBIC),
        ToTensor()
    ])

class TraindataLoad(Dataset):
    def __init__(self,img_folder,scale_factor,cropsize):
        super(TraindataLoad,self).__init__()
        self.imgs=[join(img_folder,x) for x in listdir(img_folder) if is_img(x)]
        self.crop_size=valid_crop_size(cropsize,scale_factor)
        self.hr_transform=hr_transform(self.crop_size)
        self.lr_transform=lr_transform(self.crop_size,scale_factor)
    
    def __getitem__(self,index):
        img=Image.open(self.imgs[index])
        hr_img=self.hr_transform(img)
        lr_img=self.lr_transform(hr_img)
        return lr_img,hr_img
    def __len__(self):
        return len(self.imgs)



        