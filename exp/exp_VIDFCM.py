from data.data_loader import Dataset_MTS
from exp.exp_basic import Exp_Basic
from models.VIFCM_Cross import VIDFCM
import pandas as pd
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import time
import json
import pickle
from pynvml import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings('ignore')



def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            pass
            #print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')
    return total_num

class Exp_VIDFCM(Exp_Basic):
    def __init__(self, args):
        super(Exp_VIDFCM, self).__init__(args)
    
    def _build_model(self):
        self.args.pile = int((self.args.pre_len/(self.args.grain_seq_len-self.args.grain_step))+1)
        model = VIDFCM(
            self.args.order,
            self.args.grain_seq_len,
            self.args.grain_step,
            self.args.grain_step_start,
            self.args.col,
            self.args.batch_size,
            self.args.pile,
            self.args.dropout
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        print_model_parameters(model, only_num=False)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size;
        data_set = Dataset_MTS(
            grain_seq_len=args.grain_seq_len,
            grain_step=args.grain_step,
            grain_step_start=args.grain_step_start,
            pile=args.pile,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            data_split = args.data_split,
            pre_len = args.pre_len
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,

        )

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                batch_y_decoder = torch.zeros(self.args.batch_size, self.args.col, self.args.grain_seq_len)
                enc_out_final, outputs, WW, batch_x_encoder = self._process_one_batch(
                    vali_data, batch_x, batch_y_decoder)
                pre=batch_x_encoder[:,1:,:,self.args.grain_step:].clone().to('cuda')
                true = batch_y[:,1:,self.args.grain_step:,:].clone().transpose(2, 3).float().to('cuda')
                loss = criterion(pre.reshape(pre.shape[0]*pre.shape[1], pre.shape[2], pre.shape[3]), true.reshape(true.shape[0]*true.shape[1],true.shape[2],true.shape[3]))
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                if batch_y.shape[0]!=self.args.batch_size:
                    print(iter_count)
                    print(epoch)
                    print(i)
                batch_y_decoder = torch.zeros(self.args.batch_size, self.args.col, self.args.grain_seq_len)
                model_optim.zero_grad()
                enc_out_final, outputs,WW, batch_x_encoder = self._process_one_batch(
                    train_data, batch_x, batch_y_decoder)
                pre=batch_x_encoder[:, 1:, :, self.args.grain_step:].clone().to('cuda')

                true=batch_y[:,1:,self.args.grain_step:,:].clone().transpose(2, 3).float().to('cuda')


                loss = criterion(pre.reshape(pre.shape[0]*pre.shape[1], pre.shape[2], pre.shape[3]), true.reshape(true.shape[0]*true.shape[1],true.shape[2],true.shape[3]))
                train_loss.append(loss.item())
                if (i+1) % 500==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')
        matrix = WW[0]
        matrix_1 =matrix.to('cpu')
        matrix_sum=0

        for i in range(0,WW.shape[1]):
            matrix_1_tmp=np.sum(np.linalg.norm(matrix_1[i].detach(), axis=1) ** 2)
            matrix_2_tmp = np.sqrt(matrix_1_tmp)
            matrix_3_tmp = matrix_2_tmp**2
            matrix_sum=matrix_sum+matrix_3_tmp
        norm = np.sqrt(matrix_sum)
        print(norm)
        return self.model, WW

    def test(self, setting, save_pred = False, inverse = False):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                batch_y_decoder=torch.zeros(self.args.batch_size,self.args.col, self.args.grain_seq_len )
                enc_out_final, outputs, WW, batch_x_encoder  = self._process_one_batch(
                    test_data, batch_x, batch_y_decoder)
                pred = batch_x_encoder[:, 1:, :, self.args.grain_step:].clone().to('cuda')
                true = batch_y[:,1:,self.args.grain_step:,:].clone().transpose(2, 3).float().to('cuda')
                true = true.to('cpu')
                pred = pred.to('cpu')

                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size

                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())


        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num


        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        return



    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse = False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs,batch_x_encoder,WW, enc_out_final= self.model(batch_x,batch_y)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return   enc_out_final, outputs, WW, batch_x_encoder


