import torch
import torchvision
from torch import nn as nn
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float64) / 256.
    depth[depth_png == 0] = -1.
    return depth
def plot_imgs(imgs):
    num_imgs = len(imgs)
    cols = 1
    col = cols
    for i in range(num_imgs):
        plt.subplot(num_imgs,col,i+1)
        if isinstance(imgs[i],torch.Tensor):
            
            plt.imshow(imgs[i].squeeze().detach().numpy())
        else:
            plt.imshow(imgs[i].squeeze())
    plt.show()
def plot_depth_imgs():
    groundtruth_path = "E:/KITTI/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02"
    raw_path = "E:/KITTI/data_depth_velodyne/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02"
    imgs_filename = os.listdir(groundtruth_path)
    depth_img = depth_read(os.path.join(groundtruth_path,imgs_filename[0]))
    depth_raw = depth_read(os.path.join(raw_path,imgs_filename[0]))
    print(f"read depth_img:{depth_img.shape}")
    plot_imgs([depth_img,depth_raw])
    return (depth_img,depth_raw)
class SparseConvolutionLayer(torch.nn.Module):
    def __init__(self,in_channel=1,out_channel=1,kernel_size = 3,padding = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Conv2d(in_channel,out_channel,kernel_size,padding=padding,bias=False)
        self.mean_weight = nn.AvgPool2d(kernel_size,1,padding=padding)
        self.eps = 0.001
        self.bias = torch.empty((1,out_channel,1,1),requires_grad=True)
        self.mask_maxpool = nn.MaxPool2d(kernel_size,1,padding=padding)
        torch.nn.init.xavier_uniform_(self.bias)
    def forward(self,feature,mask):
        y = feature*mask
        y = self.weight(y)
        #Use AvgPool2d in place of mean conv2d filter
        y_mask = self.mean_weight(mask)*self.kernel_size**2
        y_mask[y_mask<self.eps] = self.eps
        y_mask = 1/y_mask
        output_feature = y*y_mask + self.bias
        output_mask = self.mask_maxpool(mask)
        return (output_feature,output_mask)
        
class SparseConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparse_conv1 = SparseConvolutionLayer(1,16,11,5)
        self.sparse_conv2 = SparseConvolutionLayer(16,16,7,3)
        self.sparse_conv3 = SparseConvolutionLayer(16,16,5,2)
        self.sparse_conv4 = SparseConvolutionLayer(16,16,3,1)
        self.sparse_conv5 = SparseConvolutionLayer(16,16,3,1)
        self.sparse_conv6 = SparseConvolutionLayer(16,1,1,0)
    def forward(self,feature,mask):
        feature1,mask1 = self.sparse_conv1(feature,mask)
        feature2,mask2 = self.sparse_conv2(feature1,mask1)
        feature3,mask3 = self.sparse_conv3(feature2,mask2)
        feature4,mask4 = self.sparse_conv4(feature3,mask3)
        feature5,mask5 = self.sparse_conv5(feature4,mask4)
        feature6,mask6 = self.sparse_conv6(feature5,mask5)
        return (feature6,mask6)

if __name__ == "__main__":
    model = SparseConv().double()
    depth_imgs,depth_raw = plot_depth_imgs()
    feature = torch.asarray(depth_raw[None,None])
    mask = torch.ones_like(feature)
    mask[feature<0] = 0
    feature_output,mask_output = model(feature,mask)
    plot_imgs([feature_output,mask_output])
    # nn.ConvTranspose2d()