import argparse
from pickletools import optimize
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from datautils import TraindataLoad
from model import Generator,Discriminator
from loss import Loss
import torch.optim as opt
import  pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument("--scale_factor",default=4)
parser.add_argument("--crop_size",default=80)
parser.add_argument("--epochs",default=100)



def main():

    # device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device=torch.device("cuda:0")
    # parsing arguments
    arg=parser.parse_args()
    crop_size=arg.crop_size
    epochs=arg.epochs
    scale_factor=arg.scale_factor

    # loading data
    trainset=TraindataLoad('data/DIV2K_train_HR',scale_factor,crop_size)
    training_data=DataLoader(dataset=trainset,batch_size=64,shuffle=True)
    gen=Generator(scale_factor).to(device)
    disc=Discriminator().to(device)
    g_loss_criteria=Loss().to(device)

    g_opt=opt.Adam(gen.parameters())
    d_opt=opt.Adam(disc.parameters())
    losses={ "gen_loss":[] ,"d_loss":[],"d_score":[],"g_score":[]}
    loss = torch.nn.BCELoss()
    for i in range(epochs+1):
        traindata=tqdm(training_data)
        gen.train()
        disc.train()
        curr_loss = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        for inp,out in traindata:
            curr_loss['batch_sizes']=inp.size(0)
            inp=Variable(inp).to(device)
            out=Variable(out).to(device)
            fake_img=gen(inp)
            d_opt.zero_grad()
            fake_out=disc(fake_img).to(device)
            real_out=disc(out).to(device)
            # print(fake_out.size(),real_out.size())
            fake=torch.zeros(inp.size(0)).to(device)
            real=torch.ones(inp.size(0)).to(device)
            # d_loss=1-real_out+fake_out
            d_loss=loss(real_out,real)+loss(fake_out,fake)
            d_loss.backward(retain_graph=True)
            d_opt.step()

            g_opt.zero_grad()
            fake_img = gen(inp)
            fake_out = disc(fake_img)
            g_loss=g_loss_criteria(fake_out,fake_img,out)
            g_loss.backward()
            g_opt.step()

            curr_loss["d_loss"]=d_loss.mean()*curr_loss["batch_sizes"]
            curr_loss["g_loss"]=g_loss.mean()*curr_loss["batch_sizes"]
            curr_loss["d_score"]=real_out.mean()*curr_loss["batch_sizes"]
            curr_loss["g_score"]=fake_out.mean()*curr_loss["batch_sizes"]
            
            traindata.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                i,epochs, curr_loss['d_loss'] / curr_loss['batch_sizes'],
                curr_loss['g_loss'] / curr_loss['batch_sizes'],
                curr_loss['d_score'] / curr_loss['batch_sizes'],
                curr_loss['g_score'] / curr_loss['batch_sizes']))

            # losses["d_loss"].append(curr_loss['d_loss'] / curr_loss['batch_sizes'])
            # losses['gen_loss'].append(curr_loss['g_loss'] / curr_loss['batch_sizes'])
            # losses['d_score'].append(curr_loss['d_score'] / curr_loss['batch_sizes'])
            # losses['g_score'].append(curr_loss['g_score'] / curr_loss['batch_sizes'])
        


        if(i % 10 ==0):
            torch.save(gen.state_dict(),"saving/gen/gen_{}.pth".format(i))
            torch.save(disc.state_dict(),"saving/disc/disc_{}.pth".format(i))
        # if(i==epochs or i==0):
        #     dataframe=pd.DataFrame(data={
        #         "D_loss":losses[d_loss],"G_loss":losses['gen_loss'],"G_score":losses['g_score'],"D-score":losses['d_score']},index=range(1,epochs+1))
        #     dataframe.to_csv("losses.csv",index_label='Epochs')

            






if __name__=="__main__":
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"
    # torch.cuda.set_device("1")
    # device=torch.device("cuda")
    # print(device)
    # device=torch.cuda.set_device(1)
    # print(device)
    main()


