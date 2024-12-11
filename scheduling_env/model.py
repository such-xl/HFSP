import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,max_seq_len,model_dim):
        # 生成固定的正余弦位置编码
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe = torch.zeros(max_seq_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
class Encoder(nn.Module):
    def __init__(self,num_embeddings,embed_dim,num_heads,dropout):
        super().__init__()
        # self.embedding = nn.Embedding(num_embeddings,embed_dim,padding_idx=0)
        self.positional_encoding = PositionalEncoding()
        self.attn = nn.MultiheadAttention(embed_dim,num_heads,dropout=dropout) 
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.GELU(),
            nn.Linear(2*embed_dim,embed_dim)
        )
        self.norm0 = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,state,key_padding_mask=None):
        """
            state: (batch_size,seq_len,embed_dim)
            key_padding_mask: (batch_size,seq_len)
        """
        # self attention
        state = state + self.positional_encoding(state.size(1),state.size(2)).to(state.device)
        state = state.transpose(0,1)
        attn_output, _ = self.attn(state,state,state,key_padding_mask=key_padding_mask)
        # post layer norm
        attn_output = attn_output.transpose(0,1)
        state = state.transpose(0,1)
        attn_output = self.norm0(state+self.dropout(attn_output))
        # feed forward
        ff_output = self.feed_forward(attn_output)
        # post layer norm
        output = self.norm1(attn_output+self.dropout(ff_output))
        return output

class Decoder(nn.Module):
    def __init__(self,machine_state_dim,embedding_dim,num_heads,dropout):
        super().__init__()
        # self.m_embedding = nn.Embedding(num_m_embed,m_embedding,padding_idx=0)
        # self.a_embedding = nn.Embedding(num_a_embed,a_embedding)
        # self.enbedding_dim = m_embedding+a_embedding
        # self.embedding_dim = m_embedding+a_embedding
        self.positional_encoding = PositionalEncoding()
        self.attn_0 = nn.MultiheadAttention(machine_state_dim,num_heads,dropout=dropout)
        self.attn_1 = nn.MultiheadAttention(embedding_dim,num_heads,dropout=dropout)
        self.feed_forward0 = nn.Sequential(
            nn.Linear(machine_state_dim,2*embedding_dim),
            nn.GELU(),
            nn.Linear(2*embedding_dim,embedding_dim)
        )
        self.feed_forward1 = nn.Sequential(
            nn.Linear(embedding_dim,2*embedding_dim),
            nn.GELU(),
            nn.Linear(2*embedding_dim,embedding_dim)
        )
        self.norm0 = nn.LayerNorm(machine_state_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,tgt,mem,tgt_padding_mask = None):
        # self attention
        tgt = tgt.clone()
        tgt_embedding = tgt
        tgt_embedding = tgt_embedding + self.positional_encoding(tgt_embedding.size(1),tgt_embedding.size(2)).to(tgt_embedding.device)
        tgt_embedding = tgt_embedding.transpose(0,1)
        mem = mem.transpose(0,1)
        attn_output, _ = self.attn_0(tgt_embedding,tgt_embedding,tgt_embedding,key_padding_mask=tgt_padding_mask)
        attn_output = attn_output.transpose(0,1)
        tgt_embedding = tgt_embedding.transpose(0,1)
        # post layer norm
        attn_output  = self.norm0(tgt_embedding + self.dropout(attn_output))

        # feed forward 0
        ff_output0 = self.feed_forward0(attn_output)
        ff_output0 = self.norm1(self.dropout(ff_output0))
        # cross attention
        ff_output0 = ff_output0.transpose(0,1)
        attn_output1, _ = self.attn_1(ff_output0,mem,mem)
        
        attn_output1 = attn_output1.transpose(0,1)
        ff_output0 =ff_output0.transpose(0,1)
        # post layer norm
        attn_output1 = self.norm1(ff_output0+self.dropout(attn_output1))

        # feed forward
        ff_output1 = self.feed_forward1(attn_output1)
        # post layer norm
        output = self.norm2(attn_output1+self.dropout(ff_output1))
        return output
class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim):
        super(MixingNetwork, self).__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents

        # Hypernetworks for producing weights and biases
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, 64 * n_agents),
            nn.GELU()
        )
        self.hyper_w_2 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU()
        )
        self.hyper_b_1 = nn.Linear(state_dim, 64)
        self.hyper_b_2 = nn.Linear(state_dim, 1)

    def forward(self, agent_qs, state):
        batch_size = state.size(0)
        state = state.view(batch_size, -1)
        # Generate weights and biases from hypernetworks
        w1 = self.hyper_w_1(state).view(batch_size, self.n_agents, 64)
        b1 = self.hyper_b_1(state).view(batch_size, 1, 64)
        w2 = self.hyper_w_2(state).view(batch_size, 64, 1)
        b2 = self.hyper_b_2(state).view(batch_size, 1, 1)

        # Compute mixing layers
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1  # First layer
        hidden = torch.relu(hidden)
        q_tot = torch.bmm(hidden, w2) + b2  # Second layer

        return q_tot.squeeze(-1)
class D3QN(nn.Module):
    def __init__(self,state_seq_len,embedding_dim,machine_state_dim,action_dim,num_heads,dropout):
        super().__init__()
        self.encoder = Encoder(state_seq_len,embedding_dim,num_heads,dropout)
        self.decoder = Decoder(machine_state_dim,embedding_dim,num_heads,dropout)
        self.mixing_network = MixingNetwork(15,state_seq_len*embedding_dim)
        self.A_net = nn.Linear(embedding_dim,action_dim)
        self.V_net = nn.Linear(embedding_dim,1)
        self.gule = nn.GELU()
    def forward(self,mem,machine_action,padding_mask=None):
        output = self.decoder(machine_action,mem,padding_mask) 
        # A Net
        A = self.gule(self.A_net(output))
        # V Net
        V = self.gule(self.V_net(output))
        Q = V + A - A.mean(dim=-1,keepdim=True)
        return A

    
