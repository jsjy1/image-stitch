"""
written start in 2022.9.23
train_3 -> light_train ->supervise_light_train
warped MS coco dataset
"""

import os
import glob
import time

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader,Dataset
from torch import nn
import torch.nn.functional
from utils import transformer,DLT_solve,get_align_image,align_merge_poson,align_merge_weight
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# python -m visdom.server


class Dataloader(Dataset):
    def __init__(self,dt_path,is_train=True):
        if is_train:
            path = os.path.join(dt_path,"train")
        else:
            path = os.path.join(dt_path, "test")

        reference_file = os.path.join(path,"input1")
        target_file = os.path.join(path,"input2")
        label_file = os.path.join(path,"shift")
        self.reference_list = glob.glob(os.path.join(reference_file,'*.jpg'))
        self.target_list = glob.glob(os.path.join(target_file,'*.jpg'))
        self.label = glob.glob(os.path.join(label_file,'*.npy'))

    def __getitem__(self, item):
        img1 = cv2.imread(self.reference_list[item])/127.5-1
        img2 = cv2.imread(self.target_list[item])/127.5-1

        img1 = cv2.resize(img1,(128,128),interpolation=cv2.INTER_NEAREST)
        img2 = cv2.resize(img2,(128,128),interpolation=cv2.INTER_NEAREST)

        img1 = np.mean(np.rollaxis(img1,2,0),axis=0,keepdims=True)
        img2 = np.mean(np.rollaxis(img2,2,0),axis=0,keepdims=True)  #归一化  resize chw  灰度化

        return np.concatenate((img1,img2),axis=0), np.load(self.label[item]).reshape(1,8)

    def __len__(self):
        return len(self.reference_list)


def l2_normalize(feature,dim=1):
    """l2-normalize at axis_1"""
    temp = torch.norm(feature,dim=dim,keepdim=True)
    temp[temp==0] = 1
    return feature/temp   #这里容易出错 如果全0元素 norm为0


def correlation2(feature1, feature2, seach_range=2):
    """
    optical flow
    written in 2022.9.25
    """
    b, c, h, w = feature1.shape
    # temp1 = torch.zeros([b, (h // seach_range) ** 2, h // seach_range, w // seach_range], dtype=torch.float32).to(device)
    temp1 = torch.zeros([b, 2, h // seach_range, w // seach_range], dtype=torch.float32).to(device)  # 偏移

    temp2 = torch.zeros([ 2, h // seach_range, w // seach_range], dtype=torch.float32).to(device)  # 0->高  1->宽
    for i in range(h // seach_range):
        temp2[0,i] = torch.tensor([i],dtype=torch.float32).to(device)
    for i in range(w // seach_range):
        temp2[1,:,i] = torch.tensor([i],dtype=torch.float32).to(device)

    for z in range(b):
        for i in range(0, h - seach_range + 1, seach_range):
            for j in range(0, w - seach_range + 1, seach_range):
                temp = nn.functional.conv2d(feature2[z].unsqueeze(0),
                                            feature1[z, :, i:i + seach_range, j:j + seach_range].unsqueeze(0),
                                            stride=seach_range)
                temp = torch.exp(temp)/torch.exp(temp).sum()  # 归一化
                temp1[z, :, i // seach_range, j // seach_range] = torch.sum(temp*temp2,dim=(2,3))
                # temp1[z, :, i // seach_range, j // seach_range] = temp  # 这一步好好想想
    # temp1 = l2_normalize(temp1)
    # temp = torch.mean(temp1, dim=1, keepdim=True)  # 最后一个0留给disturb

    return temp1-temp2


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.feature_extract_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,ceil_mode=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.feature_extract_2 = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.feature_extract_3 = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        for p in self.parameters():
            p.requires_grad = False

        # self.regression1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(65*8*8,512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512,8),
        # )
        #
        # self.regression2 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(65*8*8, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 8),
        # )
        #
        # self.regression3 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(257*16*16, 512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 8),
        # )

        self.regression1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*8*8,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,8),
        )

        self.regression2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*16*16, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 8),
        )

        self.regression3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*32*32, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 8),
        )

    def forward(self,image_pair):
        img_pair = image_pair.to(torch.float32)
        img1 = img_pair[:,0,...].unsqueeze(1)  #升维
        img2 = img_pair[:,1,...].unsqueeze(1)

    #特征金字塔
        feature1,feature2 = self.feature_extract(img1,img2)
        # vis.images(127.5*(feature1[-1][0].unsqueeze(1)+1),nrow=16,win='f1[-1]',opts={'title':'f1[-1]'})
        # vis.images(127.5*(feature1[-2][0].unsqueeze(1)+1),nrow=16,win='f1[-2]',opts={'title':'f1[-2]'})
        # vis.images(127.5*(feature1[-3][0].unsqueeze(1)+1),nrow=16,win='f1[-3]',opts={'title':'f1[-3]'})
        # print('f1: ',feature1[-1].detach().numpy().max(),' f2: ',feature1[-2].detach().numpy().max(),' f3: ',feature1[-3].detach().numpy().max())

    #计算相关性，回归偏移量
        # 1计算全局相关性
        b,c,h,w = feature1[-1].shape
        search_range = 2  #16
        cor1 = correlation2(l2_normalize(feature1[-1]), l2_normalize(feature2[-1]), search_range)
        #回归偏移量，计算单应矩阵，warp上一级特征图
        offset1 = self.regression1(cor1) * search_range   # 把图给缩小了 补偿回
        patch_size = 32
        src_p = torch.Tensor([[0,0,patch_size-1,0,0,patch_size-1,patch_size-1,patch_size-1]]).to(device)  #上一级特征图尺寸 左上 右上  左下  右下
        # src_p = torch.cat((src_p,src_p,src_p,src_p),dim=0)
        src_p = src_p.expand(b,src_p.shape[-1])
        H1 = DLT_solve(src_p,offset1/4,device=setup.device).squeeze(1)  #去掉c
        M_tensor = torch.tensor([[patch_size/ 2.0, 0., patch_size / 2.0],
                                 [0., patch_size / 2.0, patch_size / 2.0],
                                 [0., 0., 1.]]).to(device)
        M_tile = M_tensor.unsqueeze(0).expand(b, M_tensor.shape[-2], M_tensor.shape[-1])
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(b, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1])
        H1 = torch.matmul(torch.matmul(M_tile_inv, H1), M_tile)
        feature22_warp,_ = transformer(l2_normalize(feature2[-2]), H1, (patch_size,patch_size),device=setup.device)  #STN进行单应变换
        feature22_warp = feature22_warp.permute([0,3,1,2])  #stn输出维度变化了

        # 2计算局部相关性
        b, c, h, w = feature1[-2].shape
        search_range = 2 #8
        cor2 = correlation2(l2_normalize(feature1[-2]), feature22_warp, search_range)
        # 回归偏移量，计算单应矩阵，warp上一级特征图
        offset2 = self.regression2(cor2) * search_range
        patch_size = 64
        src_p = torch.Tensor([[0, 0, patch_size - 1, 0, 0, patch_size - 1, patch_size - 1,
                               patch_size - 1]]).to(device)  # 上一级特征图尺寸 左上 右上  左下  右下
        # src_p = torch.cat((src_p, src_p, src_p, src_p), dim=0)
        src_p = src_p.expand(b, src_p.shape[-1])
        H2 = DLT_solve(src_p, (offset1+offset2) / 2,device=setup.device).squeeze(1)
        M_tensor = torch.tensor([[patch_size / 2.0, 0., patch_size / 2.0],
                                 [0., patch_size / 2.0, patch_size / 2.0],
                                 [0., 0., 1.]]).to(device)
        M_tile = M_tensor.unsqueeze(0).expand(b, M_tensor.shape[-2], M_tensor.shape[-1])
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(b, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1])
        H2 = torch.matmul(torch.matmul(M_tile_inv, H2), M_tile)
        feature23_warp, _ = transformer(l2_normalize(feature2[-3]), H2, (patch_size, patch_size),device=setup.device)  # STN进行单应变换
        feature23_warp = feature23_warp.permute([0, 3, 1, 2])  # stn输出维度变化了

        # 3计算局部相关性
        b, c, h, w = feature1[-3].shape
        search_range = 2 #4
        cor3 = correlation2(l2_normalize(feature1[-3]), feature23_warp, search_range)
        # 回归偏移量，计算单应矩阵，warp上一级特征图
        offset3 = self.regression3(cor3) * search_range

    #计算最后的H
        patch_size = 128
        src_p = torch.Tensor([[0, 0, patch_size - 1, 0, 0, patch_size - 1, patch_size - 1,patch_size - 1]]).to(device)  # 上一级特征图尺寸 左上 右上  左下  右下
        # src_p = torch.cat((src_p, src_p, src_p, src_p), dim=0)
        src_p = src_p.expand(b, src_p.shape[-1])
        H1 = DLT_solve(src_p, (offset1 ) ,device=setup.device).squeeze(1)
        H2 = DLT_solve(src_p, (offset1 + offset2) ,device=setup.device).squeeze(1)
        H3 = DLT_solve(src_p, (offset1 + offset2 + offset3) ,device=setup.device).squeeze(1)
        # H3 = DLT_solve(src_p, torch.tensor([[22,-17,-7,-13,26,9,-23,-32]],dtype=torch.float32).to(device) ,device=setup.device).squeeze(1)

        M_tensor = torch.tensor([[patch_size / 2.0, 0., patch_size / 2.0],
                                 [0., patch_size / 2.0, patch_size / 2.0],
                                 [0., 0., 1.]]).to(device)
        M_tile = M_tensor.unsqueeze(0).expand(b, M_tensor.shape[-2], M_tensor.shape[-1])
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(b, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1])
        H1 = torch.matmul(torch.matmul(M_tile_inv, H1), M_tile)
        H2 = torch.matmul(torch.matmul(M_tile_inv, H2), M_tile)
        H3 = torch.matmul(torch.matmul(M_tile_inv, H3), M_tile)

        # print('offset123: ', (offset1 + offset2 + offset3).detach().numpy())

        return H1,H2,H3,offset1,offset2,offset3


    def feature_extract(self,img1,img2):
        feature1 = []
        temp1 = self.feature_extract_1(img1)
        temp2 = self.feature_extract_2(temp1)
        temp3 = self.feature_extract_3(temp2)
        feature1.append(temp1)
        feature1.append(temp2)
        feature1.append(temp3)

        feature2 = []
        temp1 = self.feature_extract_1(img2)
        temp2 = self.feature_extract_2(temp1)
        temp3 = self.feature_extract_3(temp2)
        feature2.append(temp1)
        feature2.append(temp2)
        feature2.append(temp3)

        return feature1,feature2


class Myloss(nn.Module):

    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self,img_pair,H1,H2,H3):
        #求的warp后的图
        b = img_pair.shape[0]
        img_pair = img_pair.to(torch.float32)
        img1 = img_pair[:, 0, ...]
        img2 = img_pair[:, 1, ...].unsqueeze(1)

        patch_size = 128
        img2_1 = transformer(img2, H1, (patch_size, patch_size),device=setup.device)[0].squeeze()
        img2_2 = transformer(img2, H2, (patch_size, patch_size),device=setup.device)[0].squeeze()
        img2_3 = transformer(img2, H3, (patch_size, patch_size),device=setup.device)[0].squeeze()

        E = torch.ones(img2.shape,dtype=torch.float32).to(device)
        E1 = transformer(E, H1, (patch_size, patch_size),device=setup.device)[0].squeeze() #* img1     #mask*图像  前面相当于warp出掩膜 即重叠部分
        E2 = transformer(E, H2, (patch_size, patch_size),device=setup.device)[0].squeeze() #* img1
        E3 = transformer(E, H3, (patch_size, patch_size),device=setup.device)[0].squeeze() #* img1

        decimal = 1  # 防止/0
                    #100是缩放
        # loss_warp = b/100*(16 * patch_size ** 2 / (torch.sum(E1) + decimal) + 16 * patch_size ** 2 / (torch.sum(E2) + decimal) + 16 * patch_size ** 2 / (torch.sum(E3) + decimal))

        warp1 =  patch_size ** 2 / (torch.sum(E1) + decimal) *b
        warp2 =  patch_size ** 2 / (torch.sum(E2) + decimal) *b
        warp3 =  patch_size ** 2 / (torch.sum(E3) + decimal) *b

        E1 = E1 * img1
        E2 = E2 * img1
        E3 = E3 * img1

        loss_content = 16*torch.mean(torch.abs(img2_1-E1)) + 4*torch.mean(torch.abs(img2_2-E2)) + torch.mean(torch.abs(img2_3-E3))    #让数字大一点  不用mean了
        loss_content_warp = 16*torch.mean(torch.abs(img2_1-E1))* warp1 + 4*torch.mean(torch.abs(img2_2-E2))*warp2 + 1*torch.mean(torch.abs(img2_3-E3))*warp3
        # print(loss_content.cpu().detach().numpy(),"    ",loss_warp.cpu().detach().numpy(),end='  ')
        # print(loss_content.cpu().detach().numpy(),"    ",loss_content_warp.cpu().detach().numpy(),end='  ')
        # return  loss_content + (loss_warp if setup.warp_loss else 0)

        return  loss_content_warp if setup.warp_loss else loss_content


class Loss_point(nn.Module):
    def __init__(self):
        super(Loss_point, self).__init__()

    def forward(self,pre_offset,label):
        return  torch.mean((pre_offset-label)**2)


def show(img_pair,H1,H2,H3):
    # 求的warp后的图
    img_pair = img_pair.to(torch.float32)
    img1 = img_pair[:, 0, ...]
    # img1 = img_pair[0, 0, ...].unsqueeze(0)
    img2 = img_pair[:, 1, ...].unsqueeze(1)
    # img2 = img_pair[0, 1, ...].unsqueeze(0).unsqueeze(1)

    patch_size = 128
    img2_1 = transformer(img2, H1, (patch_size, patch_size),device=setup.device)[0].squeeze().unsqueeze(-1)
    img2_2 = transformer(img2, H2, (patch_size, patch_size),device=setup.device)[0].squeeze().unsqueeze(-1)
    img2_3 = transformer(img2, H3, (patch_size, patch_size),device=setup.device)[0].squeeze().unsqueeze(-1)

    img2_1 = img2_1.expand(img2_1.shape[0],img2_1.shape[1],3).detach().cpu().numpy()
    img2_2 = img2_2.expand(img2_2.shape[0],img2_2.shape[1],3).detach().cpu().numpy()
    img2_3 = img2_3.expand(img2_3.shape[0],img2_3.shape[1],3).detach().cpu().numpy()

    img2 = img2.squeeze().unsqueeze(-1)
    img2 = img2.expand(img2.shape[0],img2.shape[1],3).detach().cpu().numpy()

    img = np.concatenate((img2,img2_1,img2_2,img2_3),axis=1)

    E = torch.ones(img1.unsqueeze(0).shape, dtype=torch.float32).to(device)
    E1 = transformer(E, H1, (patch_size, patch_size),device=setup.device)[0].squeeze() * img1.squeeze()  # mask*图像  前面相当于warp出掩膜 即重叠部分
    E2 = transformer(E, H2, (patch_size, patch_size),device=setup.device)[0].squeeze() * img1.squeeze()
    E3 = transformer(E, H3, (patch_size, patch_size),device=setup.device)[0].squeeze() * img1.squeeze()
    E = np.concatenate((img1.squeeze().detach().cpu().numpy(),E1.detach().cpu().numpy(),E2.detach().cpu().numpy(),E3.detach().cpu().numpy()),axis=1)
    E = np.stack((E,E,E))
    E = np.rollaxis(E,0,3)

    img = np.concatenate((img,E),axis=0)

    img = (img+1)*127.5
    img[img<0] = 0
    img[img>255] = 255
    img = img.astype(np.uint8)

    # vis.image(np.transpose(img,(2,0,1)),win='img')

    if setup.show_img:
        cv2.imshow('a',img)
        cv2.waitKey()
        cv2.destroyAllWindows()


class Setup():
    epoch = 1000
    device = 'cpu'
    lr = 1e-4
    data_root_file = './dataset'
    istrain = False
    show_img = True                # 是否每轮展示第一张图片
    batch_size = 1
    pth_file = './5_1.pth'
    save_pth_file = './save.pth'
    warp_loss = False   # 是否使用warp损失


setup = Setup()
# vis = visdom.Visdom(env=u'test1')

if __name__ == '__main__':

    ds = Dataloader(setup.data_root_file,is_train=True)  #setup.istrain
    train_loader = DataLoader(dataset=ds,batch_size=setup.batch_size,shuffle=True,num_workers=0,drop_last=False)

    device = torch.device(setup.device)
    net = Net().to(device)
    if os.path.exists(setup.pth_file):
        net.load_state_dict(torch.load(setup.pth_file),strict=False)  #,map_location='cpu'
    net.to(device)
    myloss = Loss_point().to(device)

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = setup.lr)

    epoch = setup.epoch

    start = time.time()
    loss_list = [0]

    # for name,par in net.named_parameters():  #输出参数
    #     print(name,'   ',par.size())         #  net.regression3.named_parameters().__next__()[1].grad.numpy()

    if not setup.istrain:
        net.eval()

    for i in range(epoch):
        loss_temp = 0
        num = 0
        for img_pair,label in tqdm(train_loader):
            img_pair = img_pair.to(device)
            H1,H2,H3,offset1,offset2,offset3 = net(img_pair)
            loss = myloss( offset3, label.to(device) )

            # print('loss: ',loss.cpu().detach().numpy()) #,'    ',offset3[0].detach().cpu().numpy())

            if setup.show_img : #and loss_temp==0 :
                show(img_pair[0].unsqueeze(0).clone(),H1[0].unsqueeze(0),H2[0].unsqueeze(0),H3[0].unsqueeze(0))

            opt.zero_grad()  # 梯度清0
            if setup.istrain:

                loss.backward()     #反向传播求梯度
                opt.step()          #优化

            # print('grad1: ',net.feature_extract_1.named_parameters().__next__()[1].grad.numpy().max(),'  grad2: ',net.feature_extract_2.named_parameters().__next__()[1].grad.numpy().max(),'  grad3: ',net.feature_extract_3.named_parameters().__next__()[1].grad.numpy().max())
            # print()
            loss_temp += loss.cpu().detach().numpy()
            num +=1
        loss_temp /= num
        loss_list.append(loss_temp)
        # vis.line(Y=np.array([loss_temp]),X=np.array([i]),win='loss',update='append')
        print('epoch:%d    loss:%.5f   time:%.5f'%(i,loss_temp,time.time()-start))
        if  setup.istrain and loss_list[-1]<loss_list[-2]:  #np.abs(loss_temp)<2 and
            torch.save(net.state_dict(), setup.save_pth_file)
            a = 2
        start = time.time()