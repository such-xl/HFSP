import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

    def forward(self, x):
        # x的形状: [batch_size, seq_len, embedding_dim]
        return x + self.position_embeddings[:, : x.size(1)]

class PPONetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name and param is not None:
                    nn.init.zeros_(param)

    def init_positional_encoding(self,model):
        if hasattr(model, 'positional_encoding'):
            nn.init.normal_(model.positional_encoding, mean=0.0, std=0.02)
# 定义Actor网络 - 策略网络
class ActorNetwork(PPONetwork):
    def __init__(self, input_dim, output_dim, num_heads=4,dropout_rate=0.1):
        super(ActorNetwork, self).__init__()
        self.state_embedding = nn.Linear(input_dim, 2 * input_dim)
        self.attention = nn.MultiheadAttention(
            2 * input_dim, num_heads=num_heads, batch_first=True,dropout=dropout_rate
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12*input_dim, 6*input_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(6*input_dim, 3*input_dim),  # 输出单个值
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(3*input_dim, output_dim),
        )
        self.norm_after_embedding = nn.LayerNorm(2 * input_dim)
        self.norm_after_attention = nn.LayerNorm(2 * input_dim)
        self.positional_encoding = LearnablePositionalEncoding(2 * input_dim)
        self.apply(self.init_weights)
        self.init_positional_encoding(self.positional_encoding)
    def forward(self, x, mask=None):
        x = self.state_embedding(x)
        x = self.positional_encoding(x)
        x = self.norm_after_embedding(x)
        x = self.attention(x, x, x, key_padding_mask=mask)[0]
        x = self.norm_after_attention(x)
        x = self.mlp(x)
        x = x.masked_fill(mask, -1e9)  # 过滤无效动作
        return x

    def get_action(self, state, mask, tau=1.0, hard=False,available_actions=None):
        logits = self.forward(state, mask)

        logits = self.forward(state, mask)
    
        # if mask is not None:
        #     logits = logits.masked_fill(mask.bool(), float('-inf'))  # 过滤无效动作

        # gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20))  # 避免数值错误
        # gumbel_logits = (logits + gumbels) / tau
        # y = F.softmax(gumbel_logits, dim=-1)

        # if hard:
        #     action_index = gumbel_logits.argmax(dim=-1)
        #     y_hard = torch.zeros_like(logits).scatter_(-1, action_index.unsqueeze(-1), 1.0)
        #     y_out = y_hard - y.detach() + y  # 直通估计器
        #     action = action_index
        #     log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        # else:
        #     y_out = y
        #     dist = Categorical(y_out)
        #     action = dist.sample()
        #     log_prob = dist.log_prob(action)

        # entropy = -(y_out * torch.log(y_out + 1e-20)).sum(dim=-1)  # 计算熵

        # return action.cpu().item(), log_prob, entropy

        # # 如果提供了可用动作掩码，则应用它
        if hard:
            # 硬Gumbel-Softmax实现
            gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
            gumbel_logits = (logits + gumbels) / tau
            index = gumbel_logits.argmax(dim=-1)
            y_hard = torch.zeros_like(logits).scatter_(-1, index.unsqueeze(-1), 1.0)
            y = F.softmax(gumbel_logits, dim=-1)
            y_out = y_hard - y.detach() + y
        else:
            # 软Gumbel-Softmax实现
            gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
            y_out = F.softmax((logits + gumbels) / tau, dim=-1)
        if hard:
            action = index
            dist = Categorical(y_out)
            log_prob = dist.log_prob(action)
            return action.item(), log_prob, dist.entropy()
        else:
            # 为了与现有接口兼容，你可能需要从连续分布中取样
            dist = Categorical(y_out)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob, dist.entropy()


# 定义Critic网络 - 价值网络
class CriticNetwork(PPONetwork):
    def __init__(self, input_dim, num_heads=4,dropout_rate=0.1):
        super(CriticNetwork, self).__init__()
        # 状态嵌入层
        self.state_embedding = nn.Linear(input_dim, 2 * input_dim)
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            2 * input_dim, num_heads, batch_first=True,dropout=dropout_rate
        )
        # MLP 层
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60 * input_dim, 30 * input_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(30 * input_dim, 15 * input_dim),  # 输出单个值
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(15 * input_dim, 1),
        )
        self.norm_after_embedding = nn.LayerNorm(2 * input_dim)
        self.norm_after_attention = nn.LayerNorm(2 * input_dim)
        self.positional_encoding = LearnablePositionalEncoding(2 * input_dim)
        self.apply(self.init_weights)
        self.init_positional_encoding(self.positional_encoding)
    def forward(self, x, mask=None):
        x = self.state_embedding(x)
        x = self.positional_encoding(x)
        x = self.norm_after_embedding(x)
        x = self.attention(x, x, x, key_padding_mask=mask)[0]
        x = self.norm_after_attention(x)
        # x = x[:, -1, :]
        return self.mlp(x)