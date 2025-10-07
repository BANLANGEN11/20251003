import argparse
from exp.exp_VIDFCM import Exp_VIDFCM
from utils.tools import string_split
import torch
import numpy as np
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.cudnn_enabled = False


torch.cuda.empty_cache()


if torch.cuda.is_available():
    print('PyTorch can use GPU.')
    device = torch.device('cuda')
else:
    print('PyTorch cannot use GPU.')
    device = torch.device('cpu')


parser = argparse.ArgumentParser(description='VIDFCM')
parser.add_argument('--data', type=str, required=True, default='steel', help='data')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='steel.csv', help='data file')
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2',help='train/val/test split, can be ratio or number')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=2, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.002, help='optimizer initial learning rate')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument('--grain_seq_len', type=int, required=True, help='The number of Granulation length')
parser.add_argument('--grain_step_start', type=int, required=True, help='The number of Granulation length')
parser.add_argument('--grain_step', type=int, required=True, help='The number of Granulation length')
parser.add_argument('--col', type=int, default=7,help='The number of columns of the variable')
parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)
parser.add_argument('--order', type=int, default=1,help='The number of order')
parser.add_argument('--pre_len', type=int, default=168,help='pre long')
parser.add_argument('--num_fig', type=int, default=2,help='the number of var')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)


args.data_split = string_split(args.data_split)

print('Args in experiment:')
print(args)

Exp = Exp_VIDFCM

for ii in range(args.itr):
    setting = 'VIFCM_{}_itr{}'.format(args.data,ii)

    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)

    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, args.save_pred)




