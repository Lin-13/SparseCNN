import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
import os
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