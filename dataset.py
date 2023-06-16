import torch
import numpy as np
import torchvision
import torch.nn as nn
import os
import utils
from torch.utils.data import Dataset
class KITTIDataset(Dataset):
    '''
    file:str,"train" or "val"
    dataset_imgs<=0:read all file in dataset
    '''
    def __init__(self,dataset_dir = "E:/KITTI",num=2000,file = "train"):
        self.dataset_dir = dataset_dir
        self.dataset_type = file
        
        sync_filedict = os.listdir(os.path.join(self.dataset_dir,"data_depth_annotated",self.dataset_type))
        #Map: sync file name -> imgs in file imgs02
        self.num_imgs_sync = {}
        #Map:index->sync file name
        self.index_threshold_sync = {}
        self.index_threshold = []
        num_imgs = 0
        for sync in sync_filedict:
            # groundtruth_path = f"E:/KITTI/data_depth_annotated/{file}/{sync}/proj_depth/groundtruth/image_02"
            raw_path = f"{self.dataset_dir}/data_depth_velodyne/{file}/{sync}/proj_depth/velodyne_raw/image_02"
            imgs_filename = os.listdir(raw_path)
            self.num_imgs_sync[sync] = len(imgs_filename)
            num_imgs +=len(imgs_filename)
            self.index_threshold_sync[num_imgs] = sync
            self.index_threshold.append(num_imgs)
            if num_imgs>=num and num>0:
                break
        self.len = num if num >0 and num < self.index_threshold[-1] else self.index_threshold[-1]
        self.current_state = ()
    def __iter__(self):
        return self
    def __getitem__(self,index):
        raw_path = str()
        groundtruth_path = str()
        index_begin = 0
        sync = str()
        for index_threshold in self.index_threshold:
            if index < index_threshold:
                sync = self.index_threshold_sync[index_threshold]
                raw_path = f"{self.dataset_dir}/data_depth_velodyne/{self.dataset_type}/{sync}/proj_depth/velodyne_raw/image_02"
                groundtruth_path = f"{self.dataset_dir}/data_depth_annotated/{self.dataset_type}/{sync}/proj_depth/groundtruth/image_02"
                # print(sync)
                break
            index_begin=index_threshold
        imgs_filename = os.listdir(raw_path)
        #Debug log
        # print(f"index:{index},index_begin:{index_begin}")
        # print(f"index:{index},index_begin:{index_begin},sync:{sync},img:{imgs_filename[index-index_begin]}")
        feature = utils.depth_read(os.path.join(raw_path,imgs_filename[index-index_begin]))
        groundtruth = utils.depth_read(os.path.join(groundtruth_path,imgs_filename[index-index_begin]))
        self.current_state = (sync,imgs_filename[index-index_begin])
        #crop
        return (torch.asarray(feature).to(torch.float32)[0:370,0:1224][None],
                torch.asarray(groundtruth).to(torch.float32)[0:370,0:1224][None])
    def __len__(self):
        return self.len
if __name__=="__main__":
    # dataset_dir = "E:/KITTI"
    # train_filedict = os.listdir(os.path.join(dataset_dir,"data_depth_annotated\\train"))
    # print(train_filedict[0])
    # groundtruth_path = f"E:/KITTI/data_depth_annotated/train/{train_filedict[0]}/proj_depth/groundtruth/image_02"
    # raw_path = f"E:/KITTI/data_depth_velodyne/train/{train_filedict[0]}/proj_depth/velodyne_raw/image_02"
    # imgs_filename = os.listdir(groundtruth_path)
    # print(len(imgs_filename))
    dataset_train = KITTIDataset(num=0,file="train")
    utils.plot_imgs(dataset_train[67])
    print(dataset_train.num_imgs_sync)
    print(dataset_train.index_threshold_sync)
    print(len(dataset_train))