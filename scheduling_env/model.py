import torch
import torch.nn as nn
import torch.nn.functional as F

class JobMachineAttentionModel(nn.Module):
    def __init__(self, job_input_dim, job_hidden_dim, machine_input_dim, machine_hidden_dim, num_heads):
        super(JobMachineAttentionModel, self).__init__()
        self.job_fc = nn.Linear(job_input_dim, job_hidden_dim)
        self.machine_fc = nn.Linear(machine_input_dim, machine_hidden_dim)
        self.relu = nn.ReLU()
        self.multihead_attention = nn.MultiheadAttention(job_hidden_dim, num_heads)

    def forward(self, job_data, machine_data, job_mask=None, machine_mask=None):
        # Transform job information
        job_data = self.job_fc(job_data)  # [batch_size, job_seq_length, job_hidden_dim]
        job_data = self.relu(job_data)  # Apply ReLU
        job_data = job_data.transpose(0, 1)  # [job_seq_length, batch_size, job_hidden_dim]

        # Transform machine information
        machine_data = self.machine_fc(machine_data)  # [batch_size, machine_seq_length, machine_hidden_dim]
        machine_data = self.relu(machine_data)  # Apply ReLU
        machine_data = machine_data.transpose(0, 1)  # [machine_seq_length, batch_size, machine_hidden_dim]

        # Multi-head attention where machine data queries job data
        attn_output, attn_output_weights = self.multihead_attention(machine_data, job_data, job_data, key_padding_mask=job_mask)
        return attn_output.transpose(0, 1), attn_output_weights  # Return to original dims for further processing

# Example usage
job_input_dim = 300
job_hidden_dim = 256
machine_input_dim = 5
machine_hidden_dim = 256
num_heads = 8

model = JobMachineAttentionModel(job_input_dim, job_hidden_dim, machine_input_dim, machine_hidden_dim, num_heads)

job = torch.randn(1, 50, 300)
machine = torch.randn(1,30,5)
output,_ = model(job,machine)
print(output)
print(output.shape)
print(_.shape)
print(_)