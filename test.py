import torch
from torchvision.transforms import ToTensor,ToPILImage
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from model import Generator
import argparse

trans_tensor = transforms.Compose([transforms.ToTensor()])
trans_PIL=transforms.Compose([transforms.ToPILImage()])

parser=argparse.ArgumentParser()
parser.add_argument("--img_path",required=True)
parser.add_argument("--model_path",required=True)

def main():
    args=parser.parse_args()
    device=torch.device('cuda:2')
    model=Generator(4).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    model_path=args.model_path
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    img_path=args.img_path
    img=Image.open(img_path)
    # img=trans_tensor(img)
    # img=Variable(ToTensor()(img), volatile=True).unsqueeze(0)
    with torch.no_grad():
        img=trans_tensor(img)
        img=Variable(img).unsqueeze(0)
        img=img.to(device)
        out=model(img)
        # out_img=ToPILImage()(out[0].data.cpu())
        out_img=trans_PIL(out[0])
        out_img.save("outputs/3.png")







if __name__=="__main__":
    main()






# import argparse
# import time

# import torch
# from PIL import Image
# from torch.autograd import Variable
# from torchvision.transforms import ToTensor, ToPILImage
# import torchvision.transforms as transforms

# from model import Generator


# parser = argparse.ArgumentParser(description='Test Single Image')
# parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
# parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
# parser.add_argument('--image_name', type=str, help='test low resolution image name')
# parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
# opt = parser.parse_args()

# UPSCALE_FACTOR = opt.upscale_factor
# TEST_MODE = True if opt.test_mode == 'GPU' else False
# IMAGE_NAME = opt.image_name
# MODEL_NAME = opt.model_name

# model = Generator(UPSCALE_FACTOR).eval()
# if TEST_MODE:
#     model.cuda()
#     model.load_state_dict(torch.load(MODEL_NAME))
# else:
#     model.load_state_dict(torch.load( MODEL_NAME, map_location=lambda storage, loc: storage))

# image = Image.open(IMAGE_NAME)
# image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
# if TEST_MODE:
#     image = image.cuda()

# start = time.clock()
# out = model(image)
# elapsed = (time.clock() - start)
# print('cost' + str(elapsed) + 's')
# out_img = ToPILImage()(out[0].data.cpu())
# out_img.save("outputs/1.png")
