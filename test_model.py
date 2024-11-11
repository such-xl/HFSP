import os
import torch
import random
from scheduling_env.training_env import TrainingEnv
from scheduling_env.agents import Agent
from scheduling_env.utils import StateNorm


def test_model():

    data_path = os.path.dirname(os.path.abspath(__file__)) + '/scheduling_env/data/'
    env = TrainingEnv(action_dim=action_dim,reward_type=reward_type,max_machine_num=machine_seq_len,max_job_num=job_seq_len)
    train_data_path = data_path +'train_data/'
    jobs_name = sorted(os.listdir(train_data_path))
    state_norm = StateNorm(machine_seq_len,machine_input_dim,job_seq_len,job_input_dim,action_dim,scale_factor)
    agent = Agent(job_input_dim,job_hidden_dim,machine_input_dim,machine_hidden_dim,action_dim,num_heads,job_seq_len,machine_seq_len,epsilon_start,epsilon_end,epsilon_decay,tau,lr,gamma,target_update,device)
    agent.load_model('models/model0.pth')
    step_done  = 0
    for i in range(num_episodes): #episodes
        job_name = random.choice(jobs_name)
        job_name = 'vla20.fjs'
        job_path = train_data_path+job_name
        decision_machines = env.reset(jobs_path=job_path)
        done = False
        while not done:
            # 序贯决策
            for machine in decision_machines:
                machine_state,job_state,actions = env.get_state(machine,decision_machines)
                # state预处理
                machine_padded_state,machine_mask = state_norm.machine_padding(machine_state)

                job_padded_state,job_mask,action_mask = state_norm.job_padding(job_state,len(actions))
                # 采样一个动作
                action = agent.take_action(machine_padded_state,machine_mask,job_padded_state,job_mask,actions,action_mask,step_done)
                print(action)
                # 提交动作
                env.commit_action(machine,actions,action) 
            # 执行
            next_decision_machines,reward,done = env.step(decision_machines,scale_factor) # 获取的是平分的缩放奖励
            decision_machines = next_decision_machines
            step_done += 1
        print(env.time_step)


lr = 2e-6
num_episodes = 1
job_input_dim  = 32
machine_input_dim = 4
job_hidden_dim = 32
machine_hidden_dim = 32
action_dim = 32
num_heads = 4
job_seq_len = 30
machine_seq_len = 15
gamma = 1
epsilon_start = 1
epsilon_end = 0.005
epsilon_decay = 1000
tau = 0.005
target_update = 500
buffer_size = 10_000
scale_factor = 0.01
minimal_size = 1000
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
reward_type = [0,1,2]

test_model()
