import torch
from torch import nn
import math

class Generator(nn.Module):
    def __init__(self,scaling_factor):
        super(Generator,self).__init__()
        channels=64
        factor=int(math.log(scaling_factor,2))
        self.block1=nn.Conv2d(3,channels,kernel_size=9,padding=4)
        self.block2=nn.PReLU()
        block3_1=[Residualblock(channels) for i in range(16)]
        self.block3=nn.Sequential(*block3_1)
        self.block4=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.block5=nn.BatchNorm2d(channels)
        self.block6=nn.Conv2d(channels,channels*factor**2,kernel_size=3,padding=1)
        self.block7=nn.PixelShuffle(factor)
        self.block8=nn.PReLU()
        self.block9=nn.Conv2d(channels,channels*factor**2,kernel_size=3,padding=1)
        self.block10=nn.PixelShuffle(factor)
        self.block11=nn.PReLU()
        self.block12=nn.Conv2d(channels,3,kernel_size=9,padding=4)

    def forward(self,x):
        r1=self.block1(x)
        r2=self.block2(r1)
        r3=self.block3(r2)
        r4=self.block4(r3)
        r5=self.block5(r4)
        r6=self.block6(r2+r5)
        r7=self.block7(r6)
        r8=self.block8(r7)
        r9=self.block9(r8)
        r10=self.block10(r9)
        r11=self.block11(r10)
        r12=self.block12(r11)

        return ((torch.tanh(r12)+1)/2)



class Discriminator(nn.Module):
    def __init__(self):
        k=0.2
        super(Discriminator,self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.LeakyReLU(k),

            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(k),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(k),

            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(k),

            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(k),

            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(k),

            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(k),

            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(k),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512,1024,kernel_size=1),
            nn.LeakyReLU(k),
            nn.Conv2d(1024,1,kernel_size=1)
        )

    def forward(self,x):
        batch_size = x.size(0)
        return torch.sigmoid(self.block(x).view(batch_size))



class Residualblock(nn.Module):
    def __init__(self,channels):
        super(Residualblock,self).__init__()
        # self.block1=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        # self.block2=nn.BatchNorm2d(channels)
        # self.block3=nn.PReLU()
        # self.block4=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        # self.block5=nn.BatchNorm2d(channels)
        
        self.block1=nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels,channels,3,1,1),
            nn.BatchNorm2d(channels)
        )

    def forward(self,x):
        # r=self.block1(x)
        # r=self.block2(x)
        # r=self.block3(x)
        # r=self.block4(x)
        # r=self.block5(x)
        r=self.block1(x)
        return r+x
    
    