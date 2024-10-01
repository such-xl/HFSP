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
        self.relu = nn.ReLU()
        self.on_job_attn = nn.MultiheadAttention(job_input_dim,num_heads)
        self.job_attn = nn.MultiheadAttention(job_input_dim, num_heads) 
        self.machine_fc = nn.Linear(machine_input_dim,machine_hidden_dim)

        self.machine_attn = nn.MultiheadAttention(machine_hidden_dim,num_heads)
        self.j_m_attn = nn.MultiheadAttention(job_hidden_dim,num_heads)
        self.machine_multihead_attention = nn.MultiheadAttention(machine_input_dim,num_heads)
        self.job_fc = nn.Linear(job_input_dim, job_hidden_dim)
        self.machine_fc = nn.Linear(machine_input_dim, machine_hidden_dim)
        self.multihead_attention=nn.MultiheadAttention(job_hidden_dim,num_heads)
        self.q_fc = nn.Linear(machine_hidden_dim,action_dim)
    def forward(self, wait_machine,wait_job,on_job,wait_job_mask=None,on_job_mask=None, machine_mask=None):

        # 正在执行的job data 通过一个self-attention layer
        on_job = on_job.transpose(0,1)  # [job_seq_length, batch_size, job_hidden_dim]

        on_job,_ = self.on_job_attn(on_job,on_job,on_job,key_padding_mask = on_job_mask)
        on_job = self.relu(on_job[-1:,:,:]) #取最后一个输出作为on_job embeding
        wait_job = wait_job.transpose(0,1)
        wait_job_emded = torch.cat((wait_job,on_job),dim=0) #将on_job embeding 合并到wait_job
        wait_job_emded,_ = self.job_attn(wait_job_emded,wait_job_emded,wait_job_emded,key_padding_mask=wait_job_mask)

        # Transform machine information
        wait_machine_embed = self.relu(self.machine_fc(wait_machine)) #使得machine的embed_dim和job的一致
        wait_machine_embed = wait_machine_embed.transpose(0,1)
        wait_machine_embed,_ = self.machine_attn(wait_machine_embed,wait_machine_embed,wait_machine_embed)
        result,_ = self.j_m_attn(wait_machine_embed,wait_job_emded,wait_job_emded)
        result = self.q_fc(result.transpose(0,1))
        return result
class Actora(nn.Module):
    def __init__(self,job_input_dim,machine_input_dim,job_hidden_dim,machine_hidden_dim,num_heads) -> None:
        super().__init__()
        self.on_job_attn = nn.MultiheadAttention(job_input_dim,num_heads)
        self.job_attn = nn.MultiheadAttention(job_input_dim,num_heads)
        self.machine_fc = nn.Linear(machine_input_dim,machine_hidden_dim)
    def forward(self,wait_job,on_job,wait_machine,on_job_mask=None,wait_job_mask=None):
        # on_job 正在执行的job 通过self-attention 
        pass
    '''
# Example usage
job_input_dim = 100
job_hidden_dim = 128
machine_input_dim = 5
machine_hidden_dim = 128
num_heads = 1
model = Critic()

job = torch.randn(1, 30, 100)
machine = torch.randn(1,30,5)
job_1 = torch.randn(1,30,100)
machine_1 = torch.randn(1,20,5)
output0,w0= model(job,machine)
output1,w1 = model(job_1,machine_1)
print(w0)
print(w1.shape)
'''