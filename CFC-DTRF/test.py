import torch
import torchvision.transforms as transforms
from model import zongtimox as net
import cv2
import torch.nn
import time
import os

torch.nn.Module.dump_patches = True
nets = net()
nets.to('cuda')
check=torch.load('checkpoint/checkpoint_249_epoch.pkl', weights_only=False)
nets.load_state_dict(check['model'])
nets.eval()
img_dir='output_frames'

def testimage():
    with torch.no_grad():
        
        for i in os.listdir(img_dir):
            img=cv2.imread(img_dir+'/'+i)
            img = cv2.resize(img,(256, 256))
            transform=transforms.ToTensor()
            imgs=transform(img).float()
            imgs=torch.unsqueeze(imgs,dim=0)
            imgs=torch.tensor(imgs,requires_grad=False,device='cuda')
            outimg_0, outimg_1, outimg_2=nets(imgs)
            outimg_2=outimg_2.clone().detach().requires_grad_(False)
            outimg_2=torch.squeeze(outimg_2)
            out=torch.transpose(outimg_2,0,1)
            out=torch.transpose(out,1,2).cpu().numpy()*255
            cv2.imwrite('outtt/'+i.split('.')[0]+'.'+i.split('.')[-1],out)
            print(i)

if __name__ == '__main__':
    testimage()
