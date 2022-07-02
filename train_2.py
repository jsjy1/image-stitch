
"""written start in 2022.6.23"""
import os
import glob
import time

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader,Dataset
from torch import nn
import torch.nn.functional
from tqdm import tqdm
import visdom

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
        self.reference_list = glob.glob(os.path.join(reference_file,'*.jpg'))
        self.target_list = glob.glob(os.path.join(target_file,'*.jpg'))

    def __getitem__(self, item):
        img1 = cv2.imread(self.reference_list[item])/127.5-1
        img2 = cv2.imread(self.target_list[item])/127.5-1

        img1 = cv2.resize(img1,(128,128))
        img2 = cv2.resize(img2,(128,128))

        img1 = np.mean(np.rollaxis(img1,2,0),axis=0,keepdims=True)
        img2 = np.mean(np.rollaxis(img2,2,0),axis=0,keepdims=True)  #归一化  resize chw  灰度化

        return np.concatenate((img1,img2),axis=0)

    def __len__(self):
        return len(self.reference_list)


def l2_normalize(feature,dim=1):
    """l2-normalize at axis_1"""
    temp = torch.norm(feature, dim=dim, keepdim=True)
    temp[temp == 0] = 1
    return feature / temp  # 这里容易出错 如果全0元素 norm为0


def correlation(feature1, feature2, seach_range):
    """calculate correlation matrix between feature1 and feature2(both are l2 normalized)"""
    b,c,h,w = feature1.shape
    pad_feature2 = nn.functional.pad(feature2,pad=(seach_range ,seach_range,seach_range,seach_range),mode='constant')  #默认填充0
    cor_matrix = []
    for i in range(seach_range*2+1):
        for j in range(seach_range*2+1):
            # temp = torch.sum(feature1*pad_feature2[:,:,i:i+h,j:j+w],dim=1,keepdim=True)  #从左向右遍历
            temp = torch.mean(feature1*pad_feature2[:,:,i:i+h,j:j+w],dim=1,keepdim=True)# + (i*(seach_range*2+1) +j)/(seach_range*2+1)**2/100  #从左向右遍历
            cor_matrix.append(temp)
    return  nn.LeakyReLU()( torch.cat(cor_matrix,dim=1))   #沿着通道方向拼接


#禁止cuda
def DLT_solve(src_p, off_set):
    # src_p: shape=(bs, n, 4, 2)
    # off_set: shape=(bs, n, 4, 2)
    # can be used to compute mesh points (multi-H)

    bs, _ = src_p.shape
    divide = int(np.sqrt(len(src_p[0]) / 2) - 1)
    row_num = (divide + 1) * 2

    for i in range(divide):
        for j in range(divide):

            h4p = src_p[:, [2 * j + row_num * i, 2 * j + row_num * i + 1,
                            2 * (j + 1) + row_num * i, 2 * (j + 1) + row_num * i + 1,
                            2 * (j + 1) + row_num * i + row_num, 2 * (j + 1) + row_num * i + row_num + 1,
                            2 * j + row_num * i + row_num, 2 * j + row_num * i + row_num + 1]].reshape(bs, 1, 4, 2)

            pred_h4p = off_set[:, [2 * j + row_num * i, 2 * j + row_num * i + 1,
                                   2 * (j + 1) + row_num * i, 2 * (j + 1) + row_num * i + 1,
                                   2 * (j + 1) + row_num * i + row_num, 2 * (j + 1) + row_num * i + row_num + 1,
                                   2 * j + row_num * i + row_num, 2 * j + row_num * i + row_num + 1]].reshape(bs, 1, 4,
                                                                                                              2)

            if i + j == 0:
                src_ps = h4p
                off_sets = pred_h4p
            else:
                src_ps = torch.cat((src_ps, h4p), axis=1)
                off_sets = torch.cat((off_sets, pred_h4p), axis=1)

    bs, n, h, w = src_ps.shape

    N = bs * n

    src_ps = src_ps.reshape(N, h, w)
    off_sets = off_sets.reshape(N, h, w)

    dst_p = src_ps + off_sets

    ones = torch.ones(N, 4, 1).to(device)
    # if torch.cuda.is_available():
    #     ones = ones.cuda()
    xy1 = torch.cat((src_ps, ones), 2)
    zeros = torch.zeros_like(xy1).to(device)
    # if torch.cuda.is_available():
    #     zeros = zeros.cuda()

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.pinverse(A)  #inv->pinv
    h8 = torch.matmul(Ainv, b).reshape(N, 8)

    H = torch.cat((h8, ones[:, 0, :]), 1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)
    return torch.inverse(H)


#禁止cuda
def transformer(U, theta, out_size, **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """

    def _repeat(x, n_repeats):

        rep = torch.ones([n_repeats, ]).unsqueeze(0)
        rep = rep.int()
        x = x.int()

        x = torch.matmul(x.reshape([-1,1]), rep)
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size, scale_h):

        num_batch, num_channels , height, width = im.size()

        height_f = height
        width_f = width
        out_height, out_width = out_size[0], out_size[1]

        zero = 0
        max_y = height - 1
        max_x = width - 1
        if scale_h:

            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x).to(device)
        x1 = torch.clamp(x1, zero, max_x).to(device)
        y0 = torch.clamp(y0, zero, max_y).to(device)
        y1 = torch.clamp(y1, zero, max_y).to(device)
        dim2 = torch.from_numpy( np.array(width) ).to(device)
        dim1 = torch.from_numpy( np.array(width * height) )

        base = _repeat(torch.arange(0,num_batch) * dim1, out_height * out_width).to(device)
        # if torch.cuda.is_available():
        #     dim2 = dim2.cuda()
        #     dim1 = dim1.cuda()
        #     y0 = y0.cuda()
        #     y1 = y1.cuda()
        #     x0 = x0.cuda()
        #     x1 = x1.cuda()
        #     base = base.cuda()
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # channels dim
        im = im.permute(0,2,3,1)
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(height * width * num_batch,num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(height * width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(height * width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(height * width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa*Ia+wb*Ib+wc*Ic+wd*Id

        return output

    def _meshgrid(height, width, scale_h):

        if scale_h:
            x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 1), 1, 0))
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),
                               torch.ones([1, width]))
        else:
            x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(0.0, width.float(), width), 1), 1, 0))
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height.float(), height), 1),
                               torch.ones([1, width]))


        x_t_flat = x_t.reshape((1, -1)).float()
        y_t_flat = y_t.reshape((1, -1)).float()

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0).to(device)
        # if torch.cuda.is_available():
        #     grid = grid.cuda()
        return grid

    def _transform(theta, input_dim, out_size, scale_h):
        num_batch, num_channels , height, width = input_dim.size()
        #  Changed
        theta = theta.reshape([-1, 3, 3]).float()

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, scale_h)
        grid = grid.unsqueeze(0).reshape([1,-1])
        shape = grid.size()
        grid = grid.expand(num_batch,shape[1])
        grid = grid.reshape([num_batch, 3, -1])

        T_g = torch.matmul(theta, grid)
        x_s = T_g[:,0,:]
        y_s = T_g[:,1,:]
        t_s = T_g[:,2,:]

        t_s_flat = t_s.reshape([-1])

        # smaller
        small = 1e-7
        smallers = 1e-6*(1.0 - torch.ge(torch.abs(t_s_flat), small).float())

        t_s_flat = t_s_flat + smallers
        condition = torch.sum(torch.gt(torch.abs(t_s_flat), small).float())
        # Ty changed
        x_s_flat = x_s.reshape([-1]) / t_s_flat
        y_s_flat = y_s.reshape([-1]) / t_s_flat

        input_transformed = _interpolate( input_dim, x_s_flat, y_s_flat,out_size,scale_h)

        output = input_transformed.reshape([num_batch, out_height, out_width, num_channels ])
        return output, condition

    img_w = U.size()[2]
    img_h = U.size()[1]

    scale_h = True
    output, condition = _transform(theta, U, out_size, scale_h)
    return output, condition


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

        self.regression1 = nn.Sequential(
            nn.Conv2d((16*2+1)**2,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512*16*16,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,8),
        )

        self.regression2 = nn.Sequential(
            nn.Conv2d((8*2+1)**2, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 8),
        )

        self.regression3 = nn.Sequential(
            nn.Conv2d((4*2+1)**2, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 8),
        )

        # nn.init.kaiming_normal(self.feature_extract_1.weight,mode = 'fan_in')


    def forward(self,image_pair):
        img_pair = image_pair.to(torch.float32)
        img1 = img_pair[:,0,...].unsqueeze(1)  #升维
        img2 = img_pair[:,1,...].unsqueeze(1)

    #特征金字塔
        feature1,feature2 = self.feature_extract(img1,img2)
        vis.images(127.5 * (feature1[-1][0].unsqueeze(1) + 1), nrow=16, win='f1[-1]', opts={'title': 'f1[-1]'})
        vis.images(127.5 * (feature1[-2][0].unsqueeze(1) + 1), nrow=16, win='f1[-2]', opts={'title': 'f1[-2]'})
        vis.images(127.5 * (feature1[-3][0].unsqueeze(1) + 1), nrow=16, win='f1[-3]', opts={'title': 'f1[-3]'})
    #计算相关性，回归偏移量
        # 1计算全局相关性
        b,c,h,w = feature1[-1].shape
        search_range = 16
        cor1 = correlation(l2_normalize(feature1[-1]), l2_normalize(feature2[-1]), search_range)
        #回归偏移量，计算单应矩阵，warp上一级特征图
        offset1 = self.regression1(cor1)
        patch_size = 32
        src_p = torch.Tensor([[0,0,patch_size-1,0,0,patch_size-1,patch_size-1,patch_size-1]]).to(device)  #上一级特征图尺寸 左上 右上  左下  右下
        # src_p = torch.cat((src_p,src_p,src_p,src_p),dim=0)
        src_p = src_p.expand(b,src_p.shape[-1])
        H1 = DLT_solve(src_p,offset1/4).squeeze(1)  #去掉c
        M_tensor = torch.tensor([[patch_size/ 2.0, 0., patch_size / 2.0],
                                 [0., patch_size / 2.0, patch_size / 2.0],
                                 [0., 0., 1.]]).to(device)
        M_tile = M_tensor.unsqueeze(0).expand(b, M_tensor.shape[-2], M_tensor.shape[-1])
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(b, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1])
        H1 = torch.matmul(torch.matmul(M_tile_inv, H1), M_tile)
        feature22_warp,_ = transformer(l2_normalize(feature2[-2]), H1, (patch_size,patch_size))  #STN进行单应变换
        feature22_warp = feature22_warp.permute([0,3,1,2])  #stn输出维度变化了

        # 2计算局部相关性
        b, c, h, w = feature1[-2].shape
        search_range = 8
        cor2 = correlation(l2_normalize(feature1[-2]), feature22_warp, search_range)
        # 回归偏移量，计算单应矩阵，warp上一级特征图
        offset2 = self.regression2(cor2)
        patch_size = 64
        src_p = torch.Tensor([[0, 0, patch_size - 1, 0, 0, patch_size - 1, patch_size - 1,
                               patch_size - 1]]).to(device)  # 上一级特征图尺寸 左上 右上  左下  右下
        # src_p = torch.cat((src_p, src_p, src_p, src_p), dim=0)
        src_p = src_p.expand(b, src_p.shape[-1])
        H2 = DLT_solve(src_p, (offset1+offset2) / 2).squeeze(1)
        M_tensor = torch.tensor([[patch_size / 2.0, 0., patch_size / 2.0],
                                 [0., patch_size / 2.0, patch_size / 2.0],
                                 [0., 0., 1.]]).to(device)
        M_tile = M_tensor.unsqueeze(0).expand(b, M_tensor.shape[-2], M_tensor.shape[-1])
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(b, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1])
        H2 = torch.matmul(torch.matmul(M_tile_inv, H2), M_tile)
        feature23_warp, _ = transformer(l2_normalize(feature2[-3]), H2, (patch_size, patch_size))  # STN进行单应变换
        feature23_warp = feature23_warp.permute([0, 3, 1, 2])  # stn输出维度变化了

        # 3计算局部相关性
        b, c, h, w = feature1[-3].shape
        search_range = 4
        cor3 = correlation(l2_normalize(feature1[-3]), feature23_warp, search_range)
        # 回归偏移量，计算单应矩阵，warp上一级特征图
        offset3 = self.regression3(cor3)

    #计算最后的H
        patch_size = 128
        src_p = torch.Tensor([[0, 0, patch_size - 1, 0, 0, patch_size - 1, patch_size - 1,patch_size - 1]]).to(device)  # 上一级特征图尺寸 左上 右上  左下  右下
        # src_p = torch.cat((src_p, src_p, src_p, src_p), dim=0)
        src_p = src_p.expand(b, src_p.shape[-1])
        H1 = DLT_solve(src_p, (offset1 ) ).squeeze(1)
        H2 = DLT_solve(src_p, (offset1 + offset2) ).squeeze(1)
        H3 = DLT_solve(src_p, (offset1 + offset2 + offset3) ).squeeze(1)

        M_tensor = torch.tensor([[patch_size / 2.0, 0., patch_size / 2.0],
                                 [0., patch_size / 2.0, patch_size / 2.0],
                                 [0., 0., 1.]]).to(device)
        M_tile = M_tensor.unsqueeze(0).expand(b, M_tensor.shape[-2], M_tensor.shape[-1])
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(b, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1])
        H1 = torch.matmul(torch.matmul(M_tile_inv, H1), M_tile)
        H2 = torch.matmul(torch.matmul(M_tile_inv, H2), M_tile)
        H3 = torch.matmul(torch.matmul(M_tile_inv, H3), M_tile)

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

#原来的
# class Myloss(nn.Module):
#
#     def __init__(self):
#         super(Myloss, self).__init__()
#
#     def forward(self,img_pair,H1,H2,H3):
#         #求的warp后的图
#         b = img_pair.shape[0]
#         img_pair = img_pair.to(torch.float32)
#         img1 = img_pair[:, 0, ...]
#         img2 = img_pair[:, 1, ...].unsqueeze(1)
#
#         patch_size = 128
#         img2_1 = transformer(img2, H1, (patch_size, patch_size))[0].squeeze()
#         img2_2 = transformer(img2, H2, (patch_size, patch_size))[0].squeeze()
#         img2_3 = transformer(img2, H3, (patch_size, patch_size))[0].squeeze()
#
#         E = torch.ones(img2.shape,dtype=torch.float32).to(device)
#         E1 = transformer(E, H1, (patch_size, patch_size))[0].squeeze() #* img1     #mask*图像  前面相当于warp出掩膜 即重叠部分
#         E2 = transformer(E, H2, (patch_size, patch_size))[0].squeeze() #* img1
#         E3 = transformer(E, H3, (patch_size, patch_size))[0].squeeze() #* img1
#
#         decimal = 1  # 防止溢出
#
#         loss_warp = b/100*(16 * patch_size ** 2 / (torch.sum(E1) + decimal) + 16 * patch_size ** 2 / (torch.sum(E2) + decimal) + 16 * patch_size ** 2 / (torch.sum(E3) + decimal))
#
#         E1 = E1 * img1
#         E2 = E2 * img1
#         E3 = E3 * img1
#
#         loss_content = 16*torch.mean(torch.abs(img2_1-E1)) + 4*torch.mean(torch.abs(img2_2-E2)) + torch.mean(torch.abs(img2_3-E3))    #让数字大一点  不用mean了
#
#         # print(loss_content.cpu().detach().numpy(),"    ",loss_warp.cpu().detach().numpy(),end='  ')
#         return  loss_content + (loss_warp if setup.warp_loss else 0)
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
        img2_1 = transformer(img2, H1, (patch_size, patch_size))[0].squeeze()
        img2_2 = transformer(img2, H2, (patch_size, patch_size))[0].squeeze()
        img2_3 = transformer(img2, H3, (patch_size, patch_size))[0].squeeze()

        E = torch.ones(img2.shape,dtype=torch.float32).to(device)
        E1 = transformer(E, H1, (patch_size, patch_size))[0].squeeze() #* img1     #mask*图像  前面相当于warp出掩膜 即重叠部分
        E2 = transformer(E, H2, (patch_size, patch_size))[0].squeeze() #* img1
        E3 = transformer(E, H3, (patch_size, patch_size))[0].squeeze() #* img1

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


def show(img_pair,H1,H2,H3):
    # 求的warp后的图
    img_pair = img_pair.to(torch.float32)
    img1 = img_pair[:, 0, ...]
    # img1 = img_pair[0, 0, ...].unsqueeze(0)
    img2 = img_pair[:, 1, ...].unsqueeze(1)
    # img2 = img_pair[0, 1, ...].unsqueeze(0).unsqueeze(1)

    patch_size = 128
    img2_1 = transformer(img2, H1, (patch_size, patch_size))[0].squeeze().unsqueeze(-1)
    img2_2 = transformer(img2, H2, (patch_size, patch_size))[0].squeeze().unsqueeze(-1)
    img2_3 = transformer(img2, H3, (patch_size, patch_size))[0].squeeze().unsqueeze(-1)

    img2_1 = img2_1.expand(img2_1.shape[0],img2_1.shape[1],3).detach().cpu().numpy()
    img2_2 = img2_2.expand(img2_2.shape[0],img2_2.shape[1],3).detach().cpu().numpy()
    img2_3 = img2_3.expand(img2_3.shape[0],img2_3.shape[1],3).detach().cpu().numpy()

    img2 = img2.squeeze().unsqueeze(-1)
    img2 = img2.expand(img2.shape[0],img2.shape[1],3).detach().cpu().numpy()

    img = np.concatenate((img2,img2_1,img2_2,img2_3),axis=1)

    E = torch.ones(img1.unsqueeze(0).shape, dtype=torch.float32).to(device)
    E1 = transformer(E, H1, (patch_size, patch_size))[0].squeeze() * img1.squeeze()  # mask*图像  前面相当于warp出掩膜 即重叠部分
    E2 = transformer(E, H2, (patch_size, patch_size))[0].squeeze() * img1.squeeze()
    E3 = transformer(E, H3, (patch_size, patch_size))[0].squeeze() * img1.squeeze()
    E = np.concatenate((img1.squeeze().detach().cpu().numpy(),E1.detach().cpu().numpy(),E2.detach().cpu().numpy(),E3.detach().cpu().numpy()),axis=1)
    E = np.stack((E,E,E))
    E = np.rollaxis(E,0,3)

    img = np.concatenate((img,E),axis=0)

    img = (img+1)*127.5
    img[img<0] = 0
    img[img>255] = 255
    img = img.astype(np.uint8)

    vis.image(np.transpose(img, (2, 0, 1)), win='img')

    if setup.show_img:
        cv2.imshow('a', img)
        cv2.waitKey()
        cv2.destroyAllWindows()


class Setup():
    epoch = 1000
    device = 'cpu'
    lr = 1e-4
    data_root_file = './dataset'
    istrain = False
    show_img = True                 #是否每轮展示第一张图片
    batch_size = 1
    pth_file = './2.pth'
    save_pth_file = './4.pth'
    warp_loss = False               #是否使用warp损失


setup = Setup()
vis = visdom.Visdom(env=u'test1')

if __name__ == '__main__':

    ds = Dataloader(setup.data_root_file,is_train=setup.istrain)
    train_loader = DataLoader(dataset=ds,batch_size=setup.batch_size,shuffle=True,num_workers=0,drop_last=False)

    device = torch.device(setup.device)
    net = Net().to(device)
    if os.path.exists(setup.pth_file):
        net.load_state_dict(torch.load(setup.pth_file,map_location='cpu'))
    net.to(device)
    myloss = Myloss().to(device)

    opt = torch.optim.Adam(net.parameters(),lr = setup.lr)

    epoch = setup.epoch

    start = time.time()
    loss_list = []

    # for name,par in net.named_parameters():  #输出参数
    #     print(name,'   ',par.size())         #  net.regression3.named_parameters().__next__()[1].grad.numpy()

    if not setup.istrain:
        net.eval()

    for i in range(epoch):
        loss_temp = 0
        num=0
        for img_pair in train_loader:
            img_pair = img_pair.to(device)
            H1,H2,H3,offset1,offset2,offset3 = net(img_pair)
            loss = myloss(img_pair,H1,H2,H3)
            # print(offset1.cpu().detach().numpy(),'\n',offset2.cpu().detach().numpy(),'\n',offset3.cpu().detach().numpy())
            # print(loss.cpu().detach().numpy())

            # if setup.show_img : #and loss_temp==0 :
            show(img_pair[0].unsqueeze(0).clone(),H1[0].unsqueeze(0),H2[0].unsqueeze(0),H3[0].unsqueeze(0))

            if setup.istrain:
                opt.zero_grad()     #梯度清0
                loss.backward()     #反向传播求梯度
                opt.step()          #优化

            loss_temp += loss.cpu().detach().numpy()
            num += 1
        loss_temp /= num
        loss_list.append(loss_temp)
        vis.line(Y=np.array([loss_temp]), X=np.array([i]), win='loss', update='append')
        print('epoch:%d    loss:%.5f   time:%.5f' % (i, loss_temp, time.time() - start))
        if setup.istrain and loss_list[-1] < loss_list[-2]:  # np.abs(loss_temp)<2 and
            torch.save(net.state_dict(), setup.save_pth_file)
            a = 2
        start = time.time()

    a = 1
    import matplotlib.pyplot as plt

    plt.plot(loss_list)
    plt.show()