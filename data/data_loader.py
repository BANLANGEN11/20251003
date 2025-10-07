import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import math
from utils.tools import StandardScaler


import warnings
warnings.filterwarnings('ignore')
if torch.cuda.is_available():
    device = torch.device('cuda')

class Dataset_MTS(Dataset):
    def __init__(self, grain_seq_len , grain_step, grain_step_start,pile,root_path,pre_len, data_path='steel.csv', flag='train',
                  data_split = [0.7, 0.1, 0.2], scale=True, scale_statistic=None,timeenc=1, freq='h'):

        self.timeenc = timeenc
        self.freq = freq
        self.pre_len = pre_len
        self.seq_len=grain_seq_len
        self.pile = pile
        self.step = grain_step_start
        self.step_start=grain_step_start
        self.data_list = torch.Tensor()
        self.data_list = self.data_list.to(device)
        self.grain_step=grain_step
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        distance=math.ceil(self.seq_len/self.step)
        self.scaler = StandardScaler()
        df_raw=  np.array(pd.read_csv(os.path.join(self.root_path,self.data_path)))
        df_raw_tmp= pd.read_csv(os.path.join(self.root_path, self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train , num_train + num_vali]
        border2s = [num_train, num_train + num_vali, num_train + num_vali+num_test]


        df_data = np.array(df_raw)[:, 1:].astype('float32')

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        df_raw = data
        np.savetxt('df_raw.csv', df_raw, delimiter=',', fmt='%.4f')

        self.data=torch.tensor(df_raw)
        df_raw = torch.tensor(df_raw)


        for step in tqdm(list(range(0, df_raw.shape[0] - self.seq_len-1 , self.step)),desc='Start granulation'):

            data_gra = self.data[step:step + self.seq_len, :]
            data_gra = data_gra.to(device)
            self.data_list = torch.cat((self.data_list, torch.unsqueeze(data_gra, dim=0)), dim=0)


        num_data_1=self.data_list.shape[0]
        train_num = int(num_data_1 * self.data_split[0]);
        test_num = int(num_data_1 * self.data_split[2])
        val_num = num_data_1 - train_num - test_num;

        border1s = [0, train_num+distance , train_num + val_num+distance*4 ]
        border2s = [train_num, train_num+distance + val_num, num_data_1]
        self.border1 = border1s[self.set_type]
        self.border2 = border2s[self.set_type]


        self.data_x=self.data_list[self.border1:self.border2]

        #torch.set_printoptions(profile="full")
        #data_tra_1 = data[:num_train,:]
        #data_val_1 = data[num_train:num_train+num_vali,:]
        #data_test_1 = data[num_train+num_vali:,:]
        #data_tra = self.data_list[border1s[0]:border2s[0]]
        #data_vali = self.data_list[border1s[1]:border2s[1]]
        #data_test = self.data_list[border1s[2]:border2s[2]]


    def __getitem__(self, index):
        if index>self.data_x.shape[0]-self.pile:
            index-=self.pile
        seq = self.data_x[index]
        seq_9_all = self.data_x[index:index + self.pile]
        return seq, seq_9_all

    
    def __len__(self):
        return  len(self.data_x)-self.pile

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)