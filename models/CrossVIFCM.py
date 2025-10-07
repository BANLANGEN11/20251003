import torch.nn as nn
from models.VIDFCM import  FCM_2

class Decoder(nn.Module):

    def __init__(self, grain_seq_len , batch_size,col,dropout):
        super(Decoder, self).__init__()

        self.layer1 = FCM_2(grain_seq_len , batch_size,col).to("cuda")
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,  enc_out_final, y_decoder,  batch_x_encoder, q_next ):

        y_decoder = self.dropout(y_decoder)
        v_final,  batch_x_encoder, k_next = self.layer1(q_next, y_decoder, enc_out_final, batch_x_encoder)

        return v_final,batch_x_encoder


