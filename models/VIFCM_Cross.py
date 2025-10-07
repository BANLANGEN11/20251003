import torch
import torch.nn as nn
from models.VIFCM import Encoder
from models.CrossVIFCM import Decoder
from models.VIDFCM import  FCM_1, FCM_2


class VIDFCM(nn.Module):
    def __init__(self, order,grain_seq_len , grain_step, grain_step_start ,col,batch_size, pile,dropout):
        super(VIDFCM, self).__init__()

        self.encoder = Encoder(order,grain_seq_len , batch_size,col,pile,dropout=dropout )
        self.seq_len = grain_seq_len
        self.pile = pile
        self.step = grain_step
        self.batch_size_num = batch_size
        self.col=col
        self.tmp = grain_seq_len
        self.decoder = Decoder(grain_seq_len , batch_size,col,dropout=dropout)





        
    def forward(self, x_seq,batch_y_decoder):

        batch_x_encoder = torch.zeros((self.batch_size_num, self.pile, self.col, self.tmp))
        x_seq = x_seq.transpose(1, 2)
        enc_out_final,  batch_x_encoder, q_next, WW, BIA1 = self.encoder(x_seq, batch_x_encoder)
        batch_y_decoder[:, :, :self.step] = enc_out_final[:, :, :self.step]
        FCM_2.WW_2 = WW
        FCM_2.BIA2 = BIA1
        v_final,  batch_x_encoder = self.decoder(enc_out_final, batch_y_decoder,  batch_x_encoder,q_next)

        return v_final, batch_x_encoder, WW , enc_out_final



