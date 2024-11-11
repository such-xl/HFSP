import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=30):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x + self.pe[:, :x.size(1)]    
        return x 

     
class Actor(nn.Module):
    def __init__(self, job_input_dim, job_hidden_dim, machine_input_dim, machine_hidden_dim,action_dim, num_heads):
        super().__init__()
        # self.job_pos_encoder = PositionalEncoding(job_input_dim,30)
        # self.machine_pos_encoder = PositionalEncoding(machine_input_dim,30)
    
        self.job_attn_0 = nn.MultiheadAttention(job_input_dim,num_heads)
        self.job_attn_1 = nn.MultiheadAttention(job_input_dim,num_heads)
        self.job_fc_0 = nn.Linear(job_input_dim,64)
        self.job_fc_1 = nn.Linear(64,64)
        self.machine_attn_0 = nn.MultiheadAttention(machine_input_dim,num_heads)
        self.machine_attn_1 = nn.MultiheadAttention(machine_input_dim,num_heads)
        
        self.machine_fc_0 = nn.Linear(machine_input_dim,job_input_dim)
        self.machine_fc_1 = nn.Linear(32,64)
        self.j_m_attn = nn.MultiheadAttention(64,num_heads)

        self.A_fc_0 = nn.Linear(64,64)
        self.A_fc_1 = nn.Linear(64,action_dim)

        self.V_fc_0 = nn.Linear(64,32)
        self.V_fc_1 = nn.Linear(32,1)
    def forward(self,machine_state,job_state,machine_mask,job_mask):
        """
            
        """
        ...
        # job_state = self.job_pos_encoder(job_state)
        # machine_state = self.machine_pos_encoder(machine_state)
        # shared network
        job_state = job_state.transpose(0,1)
        machine_state = machine_state.transpose(0,1)
        job_state,_ = self.job_attn_0(job_state,job_state,job_state,key_padding_mask = job_mask)
        job_state,_ = self.job_attn_1(job_state,job_state,job_state,key_padding_mask = job_mask)
        job_state = job_state.transpose(0,1)
        job_state = F.gelu(self.job_fc_0(job_state))
        job_state = self.job_fc_1(job_state).transpose(0,1)
        machine_state,_ = self.machine_attn_0(machine_state,machine_state,machine_state,key_padding_mask = machine_mask)
        machine_state,_ = self.machine_attn_1(machine_state,machine_state,machine_state,key_padding_mask = machine_mask)
        machine_state = machine_state[-1:,:,:]
        machine_state = machine_state.transpose(0,1)
        machine_state = F.gelu(self.machine_fc_0(machine_state))
        machine_state = self.machine_fc_1(machine_state).transpose(0,1) 
        machine_job_state,_ = self.j_m_attn(machine_state,job_state,job_state)
        machine_job_state = machine_job_state.transpose(0,1)
        # a network
        A = F.gelu(self.A_fc_0(machine_job_state))
        A = self.A_fc_1(A)
        # v network
        V = F.gelu(self.V_fc_0(machine_job_state))
        V = self.V_fc_1(V)
        Q = V + A - A.mean(dim=2,keepdim=True)
        return Q
class Actora(nn.Module):
    def __init__(self, job_seq_len,job_input_dim,job_hidden_dim1,job_hidden_dim2, machine_seq_len,machine_input_dim,
                machine_hidden_dim,cfc_hidden_dim1,cfc_hidden_dim2,action_dim,):
        super().__init__()
        self.flatten_layer = nn.Flatten()
        self.wait_job_fc1 = nn.Linear(job_input_dim,job_hidden_dim1) 
        self.wait_job_fc2 = nn.Linear(job_hidden_dim1,job_hidden_dim2)

        self.on_job_fc1 = nn.Linear(job_input_dim,job_hidden_dim1) 
        self.on_job_fc2 = nn.Linear(job_hidden_dim1,job_hidden_dim2)

        self.wait_machine_fc = nn.Linear(machine_input_dim,machine_hidden_dim)

        self.cfc_1 = nn.Linear(job_hidden_dim2*job_seq_len*2+machine_hidden_dim*machine_seq_len,cfc_hidden_dim1)
        self.cfc_2 = nn.Linear(cfc_hidden_dim1,cfc_hidden_dim2)
        self.q_fc = nn.Linear(cfc_hidden_dim2,action_dim)


    def forward(self, wait_machine,wait_job,on_job,mask1,mask2):
        wait_job = F.leaky_relu(self.wait_job_fc1(wait_job))
        wait_job = self.wait_job_fc2(wait_job)

        on_job = F.leaky_relu(self.on_job_fc1(on_job))
        on_job = self.on_job_fc2(on_job)

        wait_machine = F.leaky_relu(self.wait_machine_fc(wait_machine))

        wait_job = self.flatten_layer(wait_job) 
        on_job = self.flatten_layer(on_job)
        wait_machine = self.flatten_layer(wait_machine)

        all_embeding = torch.cat((wait_job,on_job,wait_machine),dim=1)

        all_embeding = F.leaky_relu(self.cfc_1(all_embeding))
        all_embeding = self.cfc_2(all_embeding)
        result = F.softplus(self.q_fc(all_embeding))

        return result