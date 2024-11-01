import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, job_input_dim, job_hidden_dim, machine_input_dim, machine_hidden_dim,action_dim,num_heads):
        super(Critic, self).__init__()
        self.relu = nn.ReLU()
        self.on_job_attn = nn.MultiheadAttention(job_input_dim,num_heads)
        self.job_attn = nn.MultiheadAttention(job_input_dim, num_heads)
        
        self.machine_fc = nn.Linear(machine_input_dim,job_input_dim)

        self.machine_attn = nn.MultiheadAttention(machine_input_dim,num_heads)
        self.j_m_attn = nn.MultiheadAttention()
        self.machine_multihead_attention = nn.MultiheadAttention(machine_input_dim,num_heads)
        self.job_fc = nn.Linear(job_input_dim, job_hidden_dim)
        self.machine_fc = nn.Linear(machine_input_dim, machine_hidden_dim)
        self.multihead_attention=nn.MultiheadAttention(job_hidden_dim,num_heads)

    def forward(self, wait_job,on_job,wait_machine, on_job_mask=None, machine_mask=None):

        # 正在执行的job data 通过一个self-attention layer
        on_job = on_job.transpose(0,1)  # [job_seq_length, batch_size, job_hidden_dim]
        on_job,_ = self.on_job_attn(on_job,on_job,on_job,on_job_mask)
        on_job = self.relu(on_job[-1,:,:]) #取最后一个输出作为on_job embeding
        wait_job = wait_job.transpose(0,1)
        wait_job = torch.cat((wait_job,on_job),dim=0) #将on_job embeding 合并到wait_job

        wait_job_emded,_ = self.job_attn(wait_job_emded,wait_job_emded,wait_job_emded)

        # Transform machine information
        wait_machine_embed = self.relu(self.machine_fc(wait_machine)) #使得machine的embed_dim和job的一致
        wait_machine_embed = wait_machine_embed.transpose(0,1)
        wait_machine_embed,_ = self.machine_attn(wait_machine_embed,wait_machine_embed,wait_machine_embed,machine_mask)
        result,_ = self.j_m_attn(wait_machine_embed,wait_job_emded,wait_job_emded)
        
        return result.transpose(0, 1), _  # Return to original dims for further processing
class Actor(nn.Module):
    def __init__(self, job_input_dim, job_hidden_dim, machine_input_dim, machine_hidden_dim,action_dim, num_heads):
        super().__init__()
        self.job_fc1 = nn.Linear(job_input_dim, job_hidden_dim)
        self.job_fc2 = nn.Linear(job_hidden_dim,machine_hidden_dim)
        self.machine_fc = nn.Linear(machine_input_dim,machine_hidden_dim)

        self.on_job_attn = nn.MultiheadAttention(machine_hidden_dim,num_heads)
        self.job_attn = nn.MultiheadAttention(machine_hidden_dim, num_heads) 
        self.machine_attn = nn.MultiheadAttention(machine_hidden_dim,num_heads)

        self.j_m_attn = nn.MultiheadAttention(machine_hidden_dim,num_heads)

        self.q_fc = nn.Linear(machine_hidden_dim,action_dim)
    def forward(self,machine_state,job_state,machine_mask,job_mask):
        """
            
        """

        on_job = F.leaky_relu(self.job_fc1(on_job))
        on_job = F.leaky_relu(self.job_fc2(on_job))
        wait_job = F.leaky_relu(self.job_fc1(wait_job))
        wait_job = F.leaky_relu(self.job_fc2(wait_job))

        # 正在执行的job data 通过一个self-attention layer
        on_job = on_job.transpose(0,1)  # [job_seq_length, batch_size, job_hidden_dim]
        on_job,_ = self.on_job_attn(on_job,on_job,on_job,key_padding_mask = on_job_mask)
        on_job = on_job[-1:,:,:] #取最后一个输出作为on_job embeding
        # 待选择的job data 通过一个self-attenation layer
        wait_job = wait_job.transpose(0,1)
        wait_job_emded = torch.cat((wait_job[0:-1,:,:],on_job),dim=0) #将on_job embeding 合并到wait_job
        wait_job_emded,_ = self.job_attn(wait_job_emded,wait_job_emded,wait_job_emded,key_padding_mask=wait_job_mask)

        # Transform machine information
        wait_machine_embed = F.leaky_relu(self.machine_fc(wait_machine)) #使得machine的embed_dim和job的一致

        wait_machine_embed = wait_machine_embed.transpose(0,1)
        wait_machine_embed,_ = self.machine_attn(wait_machine_embed,wait_machine_embed,wait_machine_embed)

        result,_ = self.j_m_attn(wait_machine_embed,wait_job_emded,wait_job_emded)

        # reward_func_2/3 奖励在[0,1] 之间，所以保证输出全是正数
        # result = F.softplus(self.q_fc(result.transpose(0,1)))
        result = self.q_fc(result.transpose(0,1))
        return result
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