import torch.nn as nn
from models.VIDFCM import FCM_1

class Encoder(nn.Module):

    def __init__(self, order,grain_seq_len ,  batch_size,col, pile,dropout):
        super().__init__()

        self.col = col
        self.grain_seq_len=grain_seq_len
        self.layer = FCM_1(order,grain_seq_len, batch_size,self.col,pile).to("cuda")
        self.dropout = nn.Dropout(p=dropout)



    def forward(self, x_seq, batch_x_encoder):

        x_seq=self.dropout(x_seq)
        v_final, batch_x_encoder, q_next, WW, BIA1  = self.layer(x_seq,  batch_x_encoder)
        return v_final,  batch_x_encoder, q_next, WW, BIA1

