import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# ======== 1. 车间环境定义 ========
class JobShopEnv:
    def __init__(self, num_jobs=4, num_machines=3, max_operations=3):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_operations = max_operations

        # 初始化作业 (每个作业包含多个工序，每个工序可以在不同机器上执行)
        self.jobs = self._generate_jobs()

        # 机器状态 (跟踪机器空闲时间)
        self.machine_times = np.zeros(num_machines)

    def _generate_jobs(self):
        """ 随机生成作业，每个作业的工序可以在不同机器上执行 """
        jobs = []
        for j in range(self.num_jobs):
            job = []
            for o in range(self.max_operations):
                machine_choices = random.sample(range(self.num_machines), k=2)  # 随机选择2台可用机器
                process_times = [random.randint(5, 15) for _ in range(2)]  # 生成加工时间
                job.append((machine_choices, process_times))
            jobs.append(job)
        return jobs

    def reset(self):
        """ 重置车间状态 """
        self.machine_times = np.zeros(self.num_machines)
        return self._get_state()

    def _get_state(self):
        """ 返回当前车间的状态，包括机器负载和作业信息 """
        job_features = np.array([len(job) for job in self.jobs])  # 作业剩余工序
        machine_features = self.machine_times  # 机器当前负载
        return np.concatenate((job_features, machine_features))

    def step(self, action):
        """ 采取调度决策（选择作业的工序执行机器） """
        job_id, op_id, machine_choice = action
        machine_idx = self.jobs[job_id][op_id][0][machine_choice]
        process_time = self.jobs[job_id][op_id][1][machine_choice]

        # 机器负载更新
        self.machine_times[machine_idx] += process_time
        self.jobs[job_id].pop(op_id)  # 移除已完成的工序

        done = all(len(job) == 0 for job in self.jobs)  # 所有作业完成
        return self._get_state(), -self.machine_times.max(), done  # 负 makespan 作为奖励

# ======== 2. 多任务学习模型 (状态表征提取) ========
class SharedRepresentation(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_dim):
        super(SharedRepresentation, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, shared_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x  # 输出通用表征

class MultiTaskScheduler(nn.Module):
    def __init__(self, input_dim, hidden_dim, shared_dim, num_machines):
        super(MultiTaskScheduler, self).__init__()

        # 共享表征层
        self.shared_representation = SharedRepresentation(input_dim, hidden_dim, shared_dim)

        # 任务特定层
        self.makespan_layer = nn.Linear(shared_dim, num_machines)  # 最小化加工时间
        self.balance_layer = nn.Linear(shared_dim, num_machines)   # 负载均衡
        self.urgent_layer = nn.Linear(shared_dim, num_machines)    # 处理紧急任务

    def forward(self, state):
        shared_features = self.shared_representation(state)
        makespan_output = self.makespan_layer(shared_features)  
        balance_output = self.balance_layer(shared_features)    
        urgent_output = self.urgent_layer(shared_features)      
        return makespan_output, balance_output, urgent_output

# ======== 3. 训练过程 ========
def train_scheduler():
    env = JobShopEnv()
    input_dim = env.num_jobs + env.num_machines  # 作业+机器状态
    hidden_dim = 64
    shared_dim = 32
    num_machines = env.num_machines

    model = MultiTaskScheduler(input_dim, hidden_dim, shared_dim, num_machines)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_episodes = 500
    batch_size = 16

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)  # 添加 batch 维度
        done = False
        total_loss = 0

        while not done:
            # 预测调度决策
            makespan_pred, balance_pred, urgent_pred = model(state)

            # 选择动作（这里简单使用 makespan 目标）
            action_machine = torch.argmax(makespan_pred, dim=1).item()
            job_id, op_id = random.choice([(j, o) for j in range(env.num_jobs) for o in range(len(env.jobs[j]))])
            action = (job_id, op_id, action_machine)

            # 执行动作
            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            # 计算损失
            target = torch.tensor([action_machine])
            loss_makespan = criterion(makespan_pred, target)
            loss_balance = criterion(balance_pred, target)
            loss_urgent = criterion(urgent_pred, target)

            # 总损失
            lambda_1, lambda_2, lambda_3 = 0.5, 0.3, 0.2
            loss = lambda_1 * loss_makespan + lambda_2 * loss_balance + lambda_3 * loss_urgent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            state = next_state

        if episode % 50 == 0:
            print(f"Episode [{episode}/{num_episodes}], Loss: {total_loss:.4f}")

    return model

# 训练调度器
trained_model = train_scheduler()

# ======== 4. 测试调度器 ========
def test_scheduler(model):
    env = JobShopEnv()
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)

    done = False
    while not done:
        makespan_pred, _, _ = model(state)
        action_machine = torch.argmax(makespan_pred, dim=1).item()
        job_id, op_id = random.choice([(j, o) for j in range(env.num_jobs) for o in range(len(env.jobs[j]))])
        action = (job_id, op_id, action_machine)
        next_state, _, done = env.step(action)
        state = torch.FloatTensor(next_state).unsqueeze(0)

    print("调度完成，最终 makespan:", env.machine_times.max())

test_scheduler(trained_model)
