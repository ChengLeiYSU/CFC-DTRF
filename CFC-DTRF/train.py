import argparse
from my_dataset import MyDataSet as data
from model import zongtimox as net
import cv2
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

ap=argparse.ArgumentParser()
ap.add_argument('-b','--batch',help='the number of batch',type=int,default='1')
ap.add_argument('-e','--epoch',help='the number of training',type=int,default='500')
ap.add_argument('-r','--resume',help='the choice of resume',type=bool,default=False)

args=vars(ap.parse_args())

vgg = models.vgg16(pretrained=True)
for param in vgg.parameters():
    param.requires_grad = False

color_layer = vgg.features[:4]
content_layer = vgg.features[:9]

def log_images(writer, img, out,ll256,gt, it):
    images_array = vutils.make_grid(img).to('cpu')
    out_array = vutils.make_grid(out * 255).to('cpu').detach()
    ll256_array = vutils.make_grid(ll256 * 255).to('cpu').detach()
    gt = vutils.make_grid(gt).to('cpu')

    writer.add_image('input', images_array, it)
    writer.add_image('out', out_array, it)
    writer.add_image('ll256',ll256_array,it)
    writer.add_image('gt',gt,it)


net=net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


da=data('data')
dataloder=DataLoader(da,batch_size=args['batch'],shuffle=True)

r_folder, g_folder = da.get_folder_paths()
print(f"rpath 对应的文件夹: {r_folder}")
print(f"gpath 对应的文件夹: {g_folder}")

optimizer=torch.optim.Adam(lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00005,params=net.parameters())
scheduler = StepLR(optimizer, step_size=150, gamma=0.5)

def proimage(im):
    images = im[image_idx, :, :, :].clone().detach().requires_grad_(False)
    image = torch.transpose(images, 0, 1)
    image = torch.transpose(image, 1, 2).cpu().numpy() * 255
    return image

writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
j = 0
h = 0

for iter in range(iteration,args['epoch']):
    print(iter)
    prob = tqdm(enumerate(dataloder),total=len(dataloder))
    if iter < 1000:
        L1 = nn.MSELoss()
    else:
        L1 = nn.L1Loss()
    
    for i,data in prob:
        gt=torch.tensor(data[0].numpy(),requires_grad=True,device='cuda')
        raw=torch.tensor(data[1].numpy(),requires_grad=True,device='cuda')
        net.to('cuda')
        color_layer = color_layer.to(device)
        content_layer = content_layer.to(device)
        
        # 提取目标图像的颜色与内容特征
        target_color_features = color_layer(gt)

        target_content_features = content_layer(gt)
        
        raw_color_features_net, raw_content_features_net, img = net(raw)
        
        # loss
        L1loss_color = L1(raw_color_features_net,target_color_features)
        L1loss_content = L1(raw_content_features_net,target_content_features)
        
        L1loss_zongtu = L1(img,gt)
        loss = L1loss_color + L1loss_content + L1loss_zongtu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新进度条显示
        prob.set_postfix(Loss=loss.item())
        h = h+1
        
        writer.add_scalar('L1loss_color', L1loss_color.item(), h)
        writer.add_scalar('L1loss_content', L1loss_content.item(), h)
        writer.add_scalar('L1loss_image', L1loss_zongtu.item(), h)
        writer.add_scalar('loss', loss.item(), h)

        c = img
        if i % 100 ==0:
            j += 100
            image_idx =random.randint(0, 0)
            predi=c[image_idx,:,:,:].clone().detach().requires_grad_(False)
            predi=torch.transpose(predi,0,1)
            predi=torch.transpose(predi,1,2).cpu().numpy()*255
            gti=proimage(gt)
            rawi=proimage(raw)
            image=np.concatenate((rawi,predi,gti),axis=1)
            image_name = 'out/sample12_out' + str(iter) + '_' + str(i) + ".png"
            if not os.path.exists('out'):
                os.makedirs('out')
            cv2.imwrite(image_name,image)
            
    if (iter + 1) % 10 == 0:    #内存不够，每十轮保留一次
        checkpoint = {"model": net.state_dict(),
                      "optimizer": optimizer,
                      "epoch": iter}

        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')
        path_checkpoint = "checkpoint/checkpoint_{}_epoch.pkl".format(iter)
        torch.save(checkpoint, path_checkpoint)

    scheduler.step()
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

writer.close()
