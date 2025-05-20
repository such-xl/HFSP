import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical
from gymnasium import spaces


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

    def forward(self, x):
        # x的形状: [batch_size, seq_len, embedding_dim]
        return x + self.position_embeddings[:, : x.size(1)]


class PositionalEncoding(nn.Module):
    """
    标准的Transformer位置编码实现
    """

    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为缓冲区，这样它就不会被认为是模型参数
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, embed_dim]
        """
        # 添加位置编码到输入中
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PPONetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name and param is not None:
                    nn.init.zeros_(param)

    def init_positional_encoding(self, model):
        if hasattr(model, "positional_encoding"):
            nn.init.normal_(model.positional_encoding, mean=0.0, std=0.02)


# 定义Actor网络 - 策略网络
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, dropout_rate=0.1):
        super(ActorNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),  # 输出单个值
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.Flatten(),
            nn.Linear(128, output_dim),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x, mask=None):
        x1 = self.mlp(x)
        logits = x1.masked_fill(mask[:, :-1], float("-inf"))
        if torch.isnan(logits).any():
            print("logits 有 NaN！")
        assert not torch.isnan(logits).any(), "logits 有 NaN！"

        return logits

    def get_action(self, state, mask=None, tau=1.0, hard=False, eval_mode=False):
        action_mask = mask[:, :-1]
        logits = self.forward(state, mask)
        if eval_mode:
            # Deterministic action selection during evaluation
            action = logits.argmax(dim=-1)
            y_out = F.softmax(logits, dim=-1)
            dist = Categorical(y_out)
            entropy = dist.entropy().mean()
            return action.item(), entropy, entropy
        if hard:
            # 硬Gumbel-Softmax实现
            gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
            gumbel_logits = (logits + gumbels) / tau
            index = gumbel_logits.argmax(dim=-1)
            y_hard = torch.zeros_like(logits).scatter_(-1, index.unsqueeze(-1), 1.0)
            y = F.softmax(gumbel_logits, dim=-1)
            y_out = y_hard - y.detach() + y
            dist = Categorical(y_out)
            log_prob = dist.log_prob(index)
            entropy = dist.entropy().mean()

            return index.item(), log_prob, entropy
        else:
            # 软Gumbel-Softmax实现
            gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
            y_out = F.softmax((logits + gumbels) / tau, dim=-1)
            dist = Categorical(y_out)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            return action.item(), log_prob, entropy


# Critic网络
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, droput_rate=0.1):
        super(CriticNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1 * input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(droput_rate),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(droput_rate),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(droput_rate),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class ActorNetwork_atten(nn.Module):
    def __init__(self, input_dim, action_dim, num_heads=4, dropout_rate=0.1):
        super(ActorNetwork_atten, self).__init__()
        # 状态嵌入层
        self.state_embedding = nn.Linear(input_dim, 2 * input_dim)
        # 多头注意力层
        self.attention_1 = nn.MultiheadAttention(
            2 * input_dim, num_heads, batch_first=True, dropout=dropout_rate
        )
        self.attention_2 = nn.MultiheadAttention(
            2 * input_dim, num_heads, batch_first=True, dropout=dropout_rate
        )
        # MLP 层
        self.feedforward = nn.Sequential(
            nn.Linear(2 * input_dim, 4 * input_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * input_dim, 8 * input_dim),  # 输出单个值
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(8 * input_dim, action_dim),
        )
        self.layer_norm = nn.LayerNorm(2 * input_dim)
        self.positional_encoding = PositionalEncoding(2 * input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        # self.apply(self.init_weights)
        # self.init_positional_encoding(self.positional_encoding)

    def forward(self, x, mask=None):
        x = self.state_embedding(x)
        x = self.positional_encoding(x)
        residual = x
        x_norm = self.layer_norm(x)
        attn_output, _ = self.attention_1(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = residual + self.dropout(attn_output)

        residual = x
        x_norm = self.layer_norm(x)
        attn_output, _ = self.attention_2(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = residual + self.dropout(attn_output)
        x = x[:, -1, :]
        logis = self.feedforward(x)
        masked_logis = logis.masked_fill(mask, float("-inf"))
        return masked_logis

    def normalize_masked_probs(self, probs, mask):
        """将概率分布重新归一化，只考虑有效动作
        mask: True代表无效/填充动作"""
        # 创建有效动作掩码 (取反，使得True代表有效)
        valid_mask = ~mask
        # 确保无效动作概率为0
        masked_probs = probs * valid_mask.float()
        # 计算有效动作概率和
        valid_probs_sum = masked_probs.sum(dim=-1, keepdim=True)
        # 重新归一化
        normalized_probs = masked_probs / (valid_probs_sum + 1e-10)
        # 确保无效动作概率为0
        normalized_probs = normalized_probs * valid_mask.float()
        return normalized_probs

    def masked_entropy(self, probs, mask):
        """只计算有效动作的熵
        mask: True代表无效/填充动作"""
        # 创建有效动作掩码
        valid_mask = ~mask
        # 确保无效动作概率为0
        masked_probs = probs * valid_mask.float()
        # 避免log(0)出现的问题
        log_probs = torch.log(masked_probs + 1e-10)
        # 只计算有效动作的熵
        entropy = -torch.sum(masked_probs * log_probs, dim=-1)
        # 根据有效动作数量进行归一化
        valid_actions = valid_mask.float().sum(dim=-1)
        normalized_entropy = entropy / torch.clamp(valid_actions, min=1.0)
        return normalized_entropy

    def get_action(self, state, mask=None, tau=1.0, hard=False, eval_mode=False):
        logits = self.forward(state, mask)
        if eval_mode:
            # Deterministic action selection during evaluation
            action = logits.argmax(dim=-1)
            y_out = F.softmax(logits, dim=-1)
            y_out = self.normalize_masked_probs(y_out, mask)
            dist = Categorical(y_out)
            entropy = self.masked_entropy(y_out, mask)
            return action.item(), entropy, entropy
        if hard:
            # 硬Gumbel-Softmax实现
            gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
            gumbel_logits = (logits + gumbels) / tau
            index = gumbel_logits.argmax(dim=-1)
            y_hard = torch.zeros_like(logits).scatter_(-1, index.unsqueeze(-1), 1.0)
            y = F.softmax(gumbel_logits, dim=-1)
            y = self.normalize_masked_probs(y, mask)
            y_out = y_hard - y.detach() + y
            dist = Categorical(y_out)
            log_prob = dist.log_prob(index)
            entropy = self.masked_entropy(y_out, mask)

            return index.item(), log_prob, entropy
        else:
            # 软Gumbel-Softmax实现
            gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
            y_out = F.softmax((logits + gumbels) / tau, dim=-1)
            y_out = self.normalize_masked_probs(y_out, mask)
            if torch.isnan(y_out).any():
                print("y_out contains NaN values")
            dist = Categorical(y_out)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = self.masked_entropy(y_out, mask)
            return action.item(), log_prob, entropy

class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    自定义特征提取网络，处理41维特征向量：
    - 前32个特征（4个作业，每个8个特征）通过注意力机制处理
    - 后9个全局特征通过线性层处理
    最后将两部分特征合并
    
    包含层归一化(LayerNorm)和Dropout以提高模型泛化能力
    """
    
    def __init__(self, observation_space, features_dim=256, dropout_rate=0.1):
        """
        初始化自定义特征提取器
        
        参数:
            observation_space: 观察空间
            features_dim: 特征输出维度
            dropout_rate: Dropout比率，用于正则化
        """
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # 确保输入是41维向量
        assert isinstance(observation_space, spaces.Box), "期望Box类型的观察空间"
        input_dim = int(np.prod(observation_space.shape))
        assert input_dim == 41, f"期望41维输入，但得到了{input_dim}维"
        
        # 设置每个作业的特征数和作业数
        self.job_feature_dim = 8
        self.num_jobs = 4
        self.global_feature_dim = 9
        self.dropout_rate = dropout_rate
        
        # 特征维度
        self.job_embedding_dim = 32  # 每个作业的嵌入维度
        self.global_hidden_dim = 32  # 全局特征的隐藏层维度
        
        # 作业特征处理 - 每个作业先通过嵌入层
        self.job_embedding = nn.Linear(self.job_feature_dim, self.job_embedding_dim)
        self.job_embedding_norm = nn.LayerNorm(self.job_embedding_dim)
        self.job_embedding_dropout = nn.Dropout(dropout_rate)
        
        # 多头注意力层处理作业间关系
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=self.job_embedding_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(self.job_embedding_dim)
        self.attention_dropout = nn.Dropout(dropout_rate)
        
        # 作业特征处理后的输出维度
        self.job_output_dim = self.job_embedding_dim * self.num_jobs // 2
        
        # 全局特征处理
        self.global_features_net = nn.Sequential(
            nn.Linear(self.global_feature_dim, self.global_hidden_dim),
            nn.LayerNorm(self.global_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
            nn.LayerNorm(self.global_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 合并后的特征处理
        self.combined_dim = self.job_output_dim + self.global_hidden_dim
        self.final_net = nn.Sequential(
            nn.Linear(self.combined_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, observations):
        """
        前向传播函数
        
        参数:
            observations: [batch_size, 41] 的张量
            
        返回:
            处理后的特征: [batch_size, features_dim] 的张量
        """
        batch_size = observations.shape[0]
        
        # 分离作业特征和全局特征
        job_features = observations[:, :self.job_feature_dim * self.num_jobs]
        global_features = observations[:, self.job_feature_dim * self.num_jobs:]
        
        # 重塑作业特征以按作业分组
        job_features = job_features.reshape(batch_size, self.num_jobs, self.job_feature_dim)
        
        # 对每个作业应用嵌入并进行层归一化和dropout
        job_embedded = self.job_embedding(job_features)  # [batch_size, num_jobs, job_embedding_dim]
        job_embedded = self.job_embedding_norm(job_embedded)
        job_embedded = F.relu(job_embedded)
        job_embedded = self.job_embedding_dropout(job_embedded)
        
        # 应用注意力机制
        attn_output, _ = self.attention(job_embedded, job_embedded, job_embedded)
        
        # 处理注意力输出 - 应用层归一化和dropout
        job_features_combined = job_embedded + self.attention_dropout(attn_output)  # 残差连接
        job_features_combined = self.attention_norm(job_features_combined)
        
        # 重塑并压缩作业特征
        job_features_flat = job_features_combined.reshape(batch_size, -1)  # [batch_size, num_jobs * job_embedding_dim]
        job_features_compressed = job_features_flat[:, :self.job_output_dim]  # 取前半部分作为输出，降维
        
        # 处理全局特征
        global_features_out = self.global_features_net(global_features)
        
        # 合并两种特征
        combined_features = torch.cat([job_features_compressed, global_features_out], dim=1)
        
        # 最终特征处理
        output_features = self.final_net(combined_features)
        
        return output_features
