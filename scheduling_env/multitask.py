import torch
import torch.nn as nn
import torch.optim as optim

# 定义 MLP 共享层（用于提取通用状态表征）
class SharedRepresentation(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SharedRepresentation, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  # 输出通用表征
        return x

# 定义任务特定层（每个任务有自己的输出）
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_dim, num_machines):
        super(MultiTaskModel, self).__init__()
        
        # 共享表征层
        self.shared_representation = SharedRepresentation(input_dim, hidden_dim, shared_dim)
        
        # 任务特定层（不同调度目标）
        self.makespan_layer = nn.Linear(shared_dim, num_machines)  # 最小化加工时间
        self.balance_layer = nn.Linear(shared_dim, num_machines)   # 负载均衡
        self.urgent_layer = nn.Linear(shared_dim, num_machines)    # 处理紧急任务

    def forward(self, state):
        # 共享层提取通用表征
        shared_features = self.shared_representation(state)
        
        # 任务特定预测
        makespan_output = self.makespan_layer(shared_features)  # 选择机器
        balance_output = self.balance_layer(shared_features)    # 选择机器
        urgent_output = self.urgent_layer(shared_features)      # 选择机器

        return makespan_output, balance_output, urgent_output