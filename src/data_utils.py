from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import random
import pickle
import os
import numpy as np
import torch
import cv2
import albumentations 
from albumentations.pytorch import ToTensorV2
import math

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(image=x)["image"] 
        
        return x, y

    def __len__(self):
        return len(self.data)

class DatasetUtils:
    def __init__(self):
        pass
    
    @staticmethod
    def get_normalize_parameters(Target_addr):
        parameters_file = f'{Target_addr}/normalize_parameters.pkl'
        with open(parameters_file, 'rb') as f:
            parameters_dict = pickle.load(f)
        mean_list = parameters_dict['mean']
        std_list = parameters_dict['std']
        return mean_list, std_list

    @staticmethod
    def get_dataset(Target_addr, phase, plate_list, severe_list, mean_list, std_list, model_name):
        numpy_data = np.concatenate([np.load(file_path) for severe in severe_list for plate in plate_list if (file_path := f'{Target_addr}{severe}_Plate_{plate}.npy') and os.path.exists(file_path)])
        numpy_target = np.concatenate([idx_s * np.ones(np.load(file_path).shape[0], dtype=np.uint8) for idx_s, severe in enumerate(severe_list) for plate in plate_list if (file_path := f'{Target_addr}{severe}_Plate_{plate}.npy') and os.path.exists(file_path)])
        label_counter = Counter(numpy_target)
        
        if model_name == 'maxvit_t':
            numpy_data = numpy_data[:, 40:1160, 40:1160, :]
        
        print(f'{phase} --- Numpy data shape: {numpy_data.shape}, ---- Numpy data mix shape: {numpy_target.shape}')
        print(label_counter)

        mean = mean_list
        std = std_list
        
        if phase == 'train':
            transform_phase = albumentations.Compose([
                albumentations.HorizontalFlip(p=0.2),
                albumentations.RandomRotate90(p=0.2),
                albumentations.VerticalFlip(p=0.2),
                albumentations.Solarize(threshold=0.95,p=0.05),
                albumentations.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.1),
                albumentations.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                
                albumentations.OneOf([
                    albumentations.MotionBlur(p=0.1),
                    albumentations.OpticalDistortion(p=0.1),
                    albumentations.GaussNoise(p=0.1),
                    albumentations.GaussianBlur(p=0.2),
                    albumentations.GaussNoise(),
                    ], p=0.1),
                
                albumentations.OneOf([
                    albumentations.RandomSunFlare(num_flare_circles_lower=1, num_flare_circles_upper=5, src_radius=250,p=0.1),
                    albumentations.RandomSnow(brightness_coeff=1.5, p=0.3)
                    ], p=0.1),
                
                albumentations.OneOf([
                    albumentations.Sharpen(),
                    albumentations.Emboss(),
                    ], p=0.1),

                albumentations.Normalize(mean, std),
                ToTensorV2(),
                ])
            
        elif phase == 'val' or phase == 'test':
            transform_phase = albumentations.Compose([
                albumentations.Normalize(mean, std),
                ToTensorV2(),
                ])

        dataset = MyDataset(numpy_data, numpy_target, transform = transform_phase)
        
        return dataset

    @staticmethod
    def get_val_csv(Target_addr, phase, plate_list, severe_list):
        for idx_s, severe in enumerate(severe_list):
            for idx_p, plate in enumerate(plate_list):
                csv_addr = Target_addr + f'{severe}_Plate_{plate}.csv'
                if os.path.exists(csv_addr):
                    try:
                        if (idx_s == 0) and (idx_p == 0):
                            well_df = pd.read_csv(csv_addr)
                        else:
                            well_df = pd.concat([well_df, pd.read_csv(csv_addr)], axis =0)
                    except:
                        well_df = pd.read_csv(csv_addr)

        print(f'well_df shape is ---- {well_df.shape}')
                
        return well_df
    
    @staticmethod
    def get_splited_list(my_list, split=4):
        n = len(my_list)
        if n >= 4:
            step = math.floor(n / split)
            result = [my_list[i:i + step] for i in range(0, step * split, step)]
            for idx, remainder in enumerate(my_list[step * split:]):
                result[idx].append(remainder)
        else:
            result = [[idx] for idx in my_list]
        return result