import torch
from torch import nn
from torchvision.models.vgg import vgg16




class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()
        vgg=vgg16(pretrained=True)
        feature_extract=nn.Sequential(*list(vgg.features)[:31]).eval()
        for i in feature_extract:
            i.requires_grad=False
        self.feature_extract=feature_extract
        self.mse_loss=nn.MSELoss()

    def forward(self,pred_labels,gen_images,true_images):
        perception_loss=self.mse_loss(self.feature_extract(gen_images),self.feature_extract(true_images))
        loss=torch.nn.BCELoss()
        real_ones=torch.ones(pred_labels.size()).to(torch.device("cuda:0"))
        adversial_loss=loss(pred_labels,real_ones)
        image_loss=self.mse_loss(gen_images,true_images)
        return image_loss+0.06*perception_loss+0.1*adversial_loss

if __name__ == "__main__":
    g_loss = Loss()
    print(g_loss)