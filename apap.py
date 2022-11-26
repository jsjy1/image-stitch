import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import numpy as np
from ransac_homography import getTransform_LSM
import torch
import copy
from tqdm import tqdm


def getNormalize2DPts(point):
    """
    :param point: [num,2]
    :return:
    """
    origin_point = copy.deepcopy(point)
    padding = np.ones(point.shape[0], dtype=np.float)
    # todo 求均值
    c = np.mean(point, axis=0)
    point[:, :2] = point[:, :2] - c[:2]
    squre = np.square(point)
    sum = np.sum(squre, axis=1)
    mean = np.mean(np.sqrt(sum))
    scale = np.sqrt(2) / mean
    t = np.array([[scale, 0, -scale * c[0]],
                  [0, scale, -scale * c[1]],
                  [0, 0, 1]], dtype=np.float)
    origin_point = np.column_stack((origin_point, padding))
    new_point = t.dot(origin_point.T)
    new_point = new_point.T[:, :2]
    return t, new_point


def getConditionerFromPts(point):
    calculate = np.expand_dims(point, 0)
    mean_pts, std_pts = cv2.meanStdDev(calculate)
    mean_pts = np.squeeze(mean_pts)
    std_pts = np.squeeze(std_pts)
    std_pts = std_pts * std_pts * point.shape[0] / (point.shape[0] - 1)
    std_pts = np.sqrt(std_pts)
    std_pts[0] = std_pts[0] + (std_pts[0] == 0)
    std_pts[1] = std_pts[1] + (std_pts[1] == 0)
    T = np.array([[np.sqrt(2) / std_pts[0], 0, (-np.sqrt(2) / std_pts[0] * mean_pts[0])],
                  [0, np.sqrt(2) / std_pts[1], (-np.sqrt(2) / std_pts[1] * mean_pts[1])],
                  [0, 0, 1]], dtype=np.float)
    return T

def apap(img1, img2, p1, p2, inline_set,H):
    """
    网格划分，求中心距离与所有p1的内点的距离，利用公式求权值，构造a阵，得到网格的局部变换阵
    输入：内点，图像     假设输入全是内点
    输出：划分网格后对应的图像变换阵
    """

    # 最终画布
    canvas_info = get_align_image(H, img1, img2)   # [[new_w,new_h],[min_x,min_y]]  包含H*src与dst的最终画布信息  min_x,min_y是画布相对于原图的偏移信息

    c1, c2 = 50,50  # 宽高 网格数
    sigma = 8.5   # 衰减权重时的系数  8.5
    gama = 0.0001  # 权重的最小值
    h, w = img1.shape
    # w, h = canvas_info[0]
    n = inline_set.shape[0]

    # 求每个网格的中心坐标  网格指画布的网格
    x_ = np.array([i*(w/c1) for i in range(c1)])   # 宽
    y_ = np.array([i*(h/c2) for i in range(c2)])   # 高
    mesh1 = np.stack([x_, y_])     # todo 最后变形  现在必须c1=c2
    x_ += w/2/c1   # 转中心坐标
    y_ += h/2/c2
    x_ += canvas_info[1][0]   # canvas转img坐标
    y_ += canvas_info[1][1]
    mesh_cor = np.array(np.meshgrid(x_, y_)).transpose(1,2,0).reshape(1,c2,c1,2)  # todo 第一层宽坐标  第二层高坐标 ... p1应该也是第一层宽 第二层高把

    # 内点集
    p1 = p1[inline_set]
    p2 = p2[inline_set]
    p1_t = p1.reshape(-1,1,1,2)  # todo 这一步得想想  最后一维表示坐标

    # 求距离与权重系数
    simgma_inv = 1. / (sigma ** 2)
    distance = np.sqrt(np.sum((mesh_cor - p1_t)**2,axis=-1))
    weight = np.maximum(gama, np.exp(- distance * simgma_inv) )  # n * c2 * c1  对于c2c1网格中心 n各内点所占的权重
    weight = weight.reshape(n, c1 * c2).transpose(1, 0)  # c1c2 * n
    # weight = np.ones([c1*c2,n])  # test

    # 构造A  n*c1c2个权重  n*2个匹配    最终形成  c1c2*2n*9
    A = np.zeros([c1*c2, n * 2, 9], dtype=np.float)
    for k in range(n):  # todo 可以向量化 少了归一化点仅仅为了求H
        A[:, 2 * k, 0] = weight[:,k] * p1[k, 0, 0]
        A[:, 2 * k, 1] = weight[:,k] * p1[k, 0, 1]
        A[:, 2 * k, 2] = weight[:,k] * 1
        A[:, 2 * k, 6] = weight[:,k] * (-p2[k, 0, 0]) * p1[k, 0, 0]
        A[:, 2 * k, 7] = weight[:,k] * (-p2[k, 0, 0]) * p1[k, 0, 1]
        A[:, 2 * k, 8] = weight[:,k] * (-p2[k, 0, 0])

        A[:, 2 * k + 1, 3] = weight[:,k] * p1[k, 0, 0]
        A[:, 2 * k + 1, 4] = weight[:,k] * p1[k, 0, 1]
        A[:, 2 * k + 1, 5] = weight[:,k] * 1
        A[:, 2 * k + 1, 6] = weight[:,k] * (-p2[k, 0, 1]) * p1[k, 0, 0]
        A[:, 2 * k + 1, 7] = weight[:,k] * (-p2[k, 0, 1]) * p1[k, 0, 1]
        A[:, 2 * k + 1, 8] = weight[:,k] * (-p2[k, 0, 1])

    # h1 = torch.svd(torch.tensor(A))[2].numpy()[:,-1]   # c1c2 * 9
    h1 =np.linalg.svd(A)[2][:,-1]   # c1c2 * 9
    h1 = h1.reshape(c2, c1, 3, 3)
    H = h1 / h1[:, :, 2:, 2:]
    # H = np.tile(H.reshape(1,1,3,3),(c2,c1,1,1)) #h1 / h1[:, :, 2:, 2:]

    # warp
    x_ = np.array([i * (w / c1) for i in range(c1 + 1)])  # 宽端点坐标
    y_ = np.array([i * (h / c2) for i in range(c2 + 1)])  # 高端点坐标
    mesh = np.array(np.meshgrid(x_, y_)).transpose(1, 2, 0)
    img = local_warp(H, img1, canvas_info, mesh)
    # img = warp_local_homography_point(canvas_info,H,img1,mesh1)

    # a=1
    return img


def transfer(H, src_point):
    target_point = H.dot(src_point.T)
    target_point = target_point / target_point[2]
    return target_point


def warp_local_homography_point(image_info, local_h, src, point):
    height = point[1]
    width = point[0]

    result_image = np.zeros([image_info[0][1], image_info[0][0], 3] if src.ndim==3 else [image_info[0][1], image_info[0][0]], dtype=np.uint8)
    for i in tqdm(range(image_info[0][1])):
        for j in range(image_info[0][0]):
            m = 0
            n = 0
            while m<point.shape[1] and i >= height[m]:
                m += 1
            while n<point.shape[1] and j >= width[n]:
                n += 1
            current_h = np.linalg.inv(local_h[m-1, n-1, :])  # 123 这里是h的逆
            target = transfer(current_h, np.array([j + image_info[1][0], i + image_info[1][1], 1]))  # 对每个点求反变换后在src的坐标  这个分网格也是对全景图而言的
            if 0 < target[0] < src.shape[1] and 0 < target[1] < src.shape[0]:
                result_image[i, j] = src[int(target[1]), int(target[0])]
    return result_image


def get_align_image(H, src, dst):
    """  src   ---H--->   dst   """
    hs, ws = src.shape[:2]
    hd, wd = dst.shape[:2]
    pts = np.array([[0, 0], [ws - 1, 0], [0, hs - 1], [ws - 1, hs - 1]])

    # 求变换后边的界限
    p = np.vstack((pts.T, np.ones([1, pts.shape[0]], dtype=pts.dtype)))
    warp = (H @ p) / ((H @ p)[2])   # todo 会报omp的错误
    min_x = int(min(0, min(warp[0])))   # 在dst和变换后的src里挑
    max_x = int(max(wd - 1, max(warp[0])))
    min_y = int(min(0, min(warp[1])))
    max_y = int(max(hd - 1, max(warp[1])))

    # 新的变换矩阵
    H = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]]) @ H  # 加上平移变换

    # 新画布的宽高
    new_w = np.around(max_x - min_x + 1, decimals=0).astype(np.int32)
    new_h = np.around(max_y - min_y + 1, decimals=0).astype(np.int32)

    return np.array([[new_w,new_h],[min_x,min_y]])   # 宽 高


def local_warp(local_h,src,canvas_info,mesh_cor):
    """
    :param local_h: 变换阵  c1*c2*3*3
    :param src: 待变换的源图
    :param canvas_info: 新画布信息  [[w,h],[min_w,min_h]]
    :param mesh_cor: 网格端点坐标  c2+1  * c1+1 *2
    """

    b = np.zeros([canvas_info[0][1],canvas_info[0][0]],np.uint8)
    local_h = np.array([[1,0,-canvas_info[1][0]],[0,1,-canvas_info[1][1]],[0,0,1]]).reshape(1,1,3,3)\
              @local_h   # 再向右向下平移到原点
    for i in tqdm(range(mesh_cor.shape[0]-1)):
        for j in range(mesh_cor.shape[1]-1):
            t = np.zeros([canvas_info[0][1],canvas_info[0][0]],np.uint8)
            t[int(mesh_cor[i,j,1]):int(mesh_cor[i+1,j,1]),int(mesh_cor[i,j,0]):int(mesh_cor[i,j+1,0])] = \
                src[int(mesh_cor[i,j,1]):int(mesh_cor[i+1,j,1]),int(mesh_cor[i,j,0]):int(mesh_cor[i,j+1,0])]

            b += cv2.warpPerspective(t, local_h[i,j], canvas_info[0])
            # cv2.imshow('c',b)
            # cv2.waitKey(1)
    return b

if __name__ == '__main__':
    a = np.arange(256).reshape(256,1).astype(np.uint8)
    a = np.tile(a,(1,256))
    h,w = a.shape
    cv2.imshow('a',a)
    src_p = np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]],np.float32)
    # dst_p = np.array([[0,h//2],[w//2,0],[w//2,h-1],[w-1,h//2]],np.float32)
    dst_p = np.array([[-50,h//2],[w//3,-50],[w//2,h-1],[w-1,h//2]],np.float32)
    H = cv2.getPerspectiveTransform(src_p,dst_p)
    b = cv2.warpPerspective(a, H, a.shape)
    cv2.imshow('b',b)
    cv2.waitKey(1)

    # local warp
    canvas_info = get_align_image(H, a, a.copy())
    # canvas_info = np.array([[w,h],[0,0]])
    c1, c2 = 10, 10  # 宽高 网格数
    h, w = a.shape
    # w, h = canvas_info[0]

    # 求每个网格的中心坐标  网格指画布的网格
    x_ = np.array([i * (w / c1) for i in range(c1+1)])  # 宽端点坐标
    y_ = np.array([i * (h / c2) for i in range(c2+1)])  # 高端点坐标
    # x_ += w / 2 / c1  # 转中心坐标
    # y_ += h / 2 / c2
    # x_ += canvas_info[1][0]  # canvas转img坐标
    # y_ += canvas_info[1][1]
    mesh_cor = np.array(np.meshgrid(x_, y_)).transpose(1, 2, 0)
    local_h = H.reshape(1,1,3,3)
    local_h = np.tile(local_h,(c2,c1,1,1))
    local_warp(local_h, a, canvas_info, mesh_cor)


    pass