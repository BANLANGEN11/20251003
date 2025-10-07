import torch
import torch.nn as nn



class FCM_1(nn.Module):
    def __init__(self,order,grain_seq_len,batch_size,col,pile):
        super(FCM_1,self).__init__()
        self.seq_len = grain_seq_len
        self.pile = pile
        self.col = col
        self.batch_num=batch_size
        for i in range(order):
            x = torch.FloatTensor(1, self.col, self.col, self.seq_len).uniform_(-1, 1)
            param = nn.Parameter(x, requires_grad=True)
            setattr(self, f"WW{i}", param)
        self.BIA1=nn.Parameter(torch.randn(self.col,self.seq_len),requires_grad=True)
        for i in range(1, pile):
            setattr(self, f'p_linear_{i}_{i + 1}', nn.Linear(self.seq_len, self.seq_len))
            setattr(self, f'q_linear_{i}_{i + 1}', nn.Linear(self.seq_len, self.seq_len))
        setattr(self, f'q_linear_{i+1}_{i+2}', nn.Linear(self.seq_len, self.seq_len))
        self.projection = nn.ModuleList([nn.Linear(self.seq_len, self.seq_len) for _ in range(pile-2)])
        self.norm = nn.ModuleList([nn.LayerNorm(self.seq_len) for _ in range(pile-2)])

    def forward(self, v_final, batch_x_encoder):

        _, F, S, T = self.WW0.shape
        A = self.batch_num
        batch_x_encoder[:, 0, :, :] = v_final
        for i in range(0, self.pile-2):
            q = v_final.float()
            p = v_final.float()
            q_linear = getattr(self, f'q_linear_{i+1}_{i + 2}')
            p_linear = getattr(self, f'p_linear_{i+1}_{i + 2}')
            q = q_linear(q)
            p = p_linear(p)
            v_tmp = v_final.repeat_interleave(self.col, dim=1)
            v_ = v_tmp.view(A, F, S, T).contiguous()
            v_final_tmp = torch.mul(self.WW0.repeat(self.batch_num,1,1,1), v_)
            p_q = torch.matmul(p, q.transpose(1, 2))
            scores = torch.softmax(p_q, dim=2)
            scores_T = scores.transpose(1, 2)
            scores_1 = scores_T.view(A, F, S, 1).contiguous()
            scores_S = scores_1.expand(A, F, S, self.seq_len)
            output = torch.mul(scores_S, v_final_tmp)
            output_final = torch.sum(output, dim=1)
            output_final = output_final + self.BIA1
            output_final = f(output_final)
            v_final = self.projection[i](output_final)
            v_final = self.norm[i](v_final)
            batch_x_encoder[:, i+1, :, :] = v_final
        q = v_final.float()
        q_next = getattr(self, f'q_linear_{i+2}_{i + 3}')(q)
        return v_final, batch_x_encoder, q_next, self.WW0, self.BIA1



class FCM_2(nn.Module):
    def __init__(self, grain_seq_len ,batch_size,col):
        super(FCM_2,self).__init__()
        self.seq_len = grain_seq_len
        self.col = col
        self.batch_num=batch_size
        self.p_linear_decoder=nn.Linear(self.seq_len, self.seq_len)
        self.norm6 = nn.LayerNorm(self.seq_len)
        self.projection_8 = nn.Linear(self.seq_len, self.seq_len, bias=True)

    def forward(self,q, p,  v_final,  batch_x_encoder):
#q改为p,k改为q
        _, F, S, T = self.WW_2.shape
        A = self.batch_num
        p = p.float().to("cuda")
        p = self.p_linear_decoder(p)
        v_tmp_7 = v_final.repeat_interleave(self.col, dim=1)
        v_7 = v_tmp_7.view(A, F, S, T).contiguous()
        v_final_tmp_8 = torch.mul(self.WW_2.repeat(self.batch_num,1,1,1), v_7)
        p_q = torch.matmul(p, q.transpose(1, 2))
        scores = torch.softmax(p_q, dim=2)
        scores_T = scores.transpose(1, 2)
        scores_1 = scores_T.view(A, F, S, 1).contiguous()
        scores_S = scores_1.expand(A, F, S, self.seq_len)
        output = torch.mul(scores_S, v_final_tmp_8)  # 哈达玛乘积
        output_final = torch.sum(output[:, :, :, :], dim=1)
        output_final = output_final + self.BIA2
        output_final = f(output_final)
        v_final_8 = output_final
        v_final_8 = self.projection_8(v_final_8)
        v_final_8 = self.norm6(v_final_8)
        batch_x_encoder[:, -1, :, :] = v_final_8

        return v_final_8,batch_x_encoder,q

def f(x):
    y = float()
    try:
        y = torch.sigmoid(x)
    except OverflowError:
        y = 1
    return y