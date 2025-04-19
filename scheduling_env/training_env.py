import random
import numpy as np
from .job import Job, JobList
import json
from .machine import Machine
from .basic_scheduling_algorithms import EDD,MS,SRO,CR
from .reward import AsyncMachineUtilizationReward
from .reward_2 import AsyncTardinessReward
np.random.seed(42)
rng_1 = np.random.default_rng(42)
rng_2 = np.random.default_rng(42)
rng_3 = np.random.default_rng(42)
rng_4 = np.random.default_rng(42)
class TrainingEnv:
    # 初始化环境
    def __init__(
        self,
        obs_dim,
        obs_len,
        state_dim,
        state_len,
        action_dim,
        max_job_num,
        job_file_path,
        rng = None
    ) -> None:
        self.obs_dim = obs_dim
        self.obs_len = obs_len
        self.state_dim = state_dim
        self.state_len = state_len
        self.action_dim = action_dim
        self.max_job_num = max_job_num
        self.machine_num = 10
        self.rng = rng or np.random.default_rng(42)
        self.reward_calculator = None

        with open(job_file_path, "r") as f:
            self.job_type = json.load(f)
            # 转换键
        for job in self.job_type:
            for process in job["process_list"]:
                # 创建新的字典来存储转换后的键
                new_process = {}
                for key, value in process.items():
                    new_process[int(key)] = value  # 将字符串键转换为整数
                # 用转换后的字典替换原字典
                job["process_list"][job["process_list"].index(process)] = new_process
        self.job_arrivals = self.create_job_arriavl_seq()
    def __init0__(
        self,obs_dim,obs_len,state_dim,state_len,action_dim,max_job_num,job_file_path) -> None:

        self.obs_dim = obs_dim
        self.obs_len = obs_len
        self.state_dim = state_dim
        self.state_len = state_len
        self.machine_num = 10
        self.action_dim = action_dim
        self.max_job_num = max_job_num
        self.file_path = job_file_path


    def create_job_arriavl_seq(self, lambda_rate=0.15):
        """
        生成指数分布的间隔时间，并取整
        """
        intervals = self.rng.exponential(
            scale=1 / lambda_rate, size=self.max_job_num - 10
        )
        intervals = np.round(intervals).astype(int)  # 取整转换为整数
        arrival_times = np.cumsum(intervals)
        arrival_times = np.insert(arrival_times, 0, [0] * 10)
        selected_jobs = [self.rng.choice(self.job_type) for _ in range(self.max_job_num)]
        arrivals = [(job, time,int(sum(sum(d.values()) / len(d) for d in job["process_list"])*self.rng.uniform(1.1,3)+time)) for job, time in zip(selected_jobs, arrival_times)]
        arrivals.sort(key=lambda x: x[1])

        return arrivals

    def insert_job(self):
        while (
            self.job_num < self.max_job_num
            and self.time_step == self.job_arrivals[self.job_num][1]
        ):
            job_info = self.job_arrivals[self.job_num][0]
            insert_job = Job(
                    id=self.job_num + 1,
                    type=job_info["type"],
                    process_num=job_info["process_num"],
                    process_list=job_info["process_list"],
                    insert_time=self.time_step,
                    due_time=self.job_arrivals[self.job_num][2]
            )
            # insert_job.due_time = int(insert_job.get_remaining_avg_time()*np.random.uniform(2, 3)+self.time_step)
            self.reward_calculator.update_job_info(insert_job.id-1,insert_job.due_time,insert_job.get_remaining_avg_time(),self.time_step)
            self.uncomplete_job.append(insert_job)
            self.job_num += 1
    def insert_job_1(self):
        while (
            self.job_num < self.max_job_num
            and self.time_step == self.job_list[self.job_num].insert_time
        ):
            self.uncomplete_job.append(self.job_list[self.job_num])
            self.job_num += 1

    def is_decision_machine(self, machine):
        """
        是否是需要做出决策的agent，当agent只能选择空闲时，则不需要做出决策
        """
        if not machine.is_idle() or machine.step_decision_made(self.time_step):
            return False
        job: Job = self.uncomplete_job.head
        while job:
            if job.is_wating_for_machine() and job.match_machine(machine.id):
                return True
            job = job.next
        return False

    def get_decsion_machines(self):
        """
        获取需要做出决策的机器
        """
        decision_machines = [
            machine for machine in self.machines if self.is_decision_machine(machine)
        ]

        self.rng.shuffle(decision_machines)
        return decision_machines  # 打乱顺序，模拟异步决策

    def get_available_jobs(self):
        """
        获取可选择的作业列表
        """
        available_jobs = []
        job = self.uncomplete_job.head
        while job:
            if job.is_wating_for_machine() and job.match_machine(
                self.current_machine.id
            ):
                available_jobs.append(job)
            job = job.next
        return available_jobs

    def reset(self):
        """
        重置环境
        reutrn:
            state: 当前job环境状态
            machine_action: 决策机器的状态
        """
        self.time_step, self.job_num = 0, 0
        self.count_actions = [0 for _ in range(self.action_dim)]
        self.machines = [Machine(i) for i in range(1, self.machine_num + 1)]
        self.makespans = [0 for _ in range(self.machine_num)]
        self.uncomplete_job = JobList()
        self.complete_job = JobList()
        self.reward_calculator = AsyncTardinessReward(self.machine_num)
        self.insert_job()
        self.pre_avg_urgency = np.mean(self.compute_urgency())
        decision_machines = self.get_decsion_machines()
        self.current_machine = decision_machines[0]
        self.available_jobs = self.get_available_jobs()
        obs_i,obs_mask = self.get_obs_i()
        global_state= self.get_global_state()
        if np.any(np.abs(obs_i) > 1):
            print("obs_i", obs_i)
            print(obs_i)
            raise ValueError("too large")
        if np.any(np.abs(global_state)>1):
            print("global_state",global_state)
            raise ValueError("too large")
        return obs_i,obs_mask, global_state,
    def reset1(self):
        """
        重置环境
        reutrn:
            state: 当前job环境状态
            machine_action: 决策机器的状态
        """
        self.time_step, self.job_num = 0, 0
        self.idle_action = 0
        self.machines = [Machine(i) for i in range(1, self.machine_num + 1)]
        self.uncomplete_job = JobList()
        self.complete_job = JobList()
        self.reward_calculator = AsyncMachineUtilizationReward(self.machine_num)
        jobs = None
        self.job_list = []
        with open(self.file_path, "r") as f:
            jobs = json.load(f)
        for i,job in enumerate(jobs,1):
            process = []
            for machine_id,process_time in zip(job["machine"],job["process"]):
                process.append({machine_id+1:process_time})
            self.job_list.append(Job(i,i,len(job["process"]),process,int(job["insert_time"]),int(job["due_time"])))
        self.job_list.sort(key=lambda x: x.insert_time)
        self.insert_job()
        decision_machines = self.get_decsion_machines()
        self.current_machine = decision_machines[0]
        self.available_jobs = self.get_available_jobs()
        obs_i, obs_mask = self.get_obs_i()
        global_state = self.get_global_state()

        return obs_i, obs_mask,global_state
    def get_global_state(self):
        """
        获取全局状态
        """
        urgency = self.compute_urgency()
        if len(urgency)==0:
            raise ValueError("urgency is empty")
        global_state = [
            np.mean(urgency),
            np.std(urgency),
            np.max(urgency),
            np.min(urgency),
            self.pre_avg_urgency,
            0
        ]
        return global_state

    def run(self):
        """
        所有忙碌agent和job更新一个time_step,使得必产生空闲机器
        在内添加随机时间
        """
        # 更新one timestep时序
        min_run_timestep = 1 # 方便后续引入机器故障等随机事件
        for machine in self.machines:
            if not machine.is_running():
                continue
            job = machine.job

            machine.run(min_run_timestep, self.time_step)

        self.time_step += min_run_timestep

        if self.job_num < self.max_job_num:
            self.insert_job()
        job: Job = self.uncomplete_job.head
        while job:
            machine = job.machine
            next_job = job.next
            self.reward_calculator.update_job_info(job.id-1,job.due_time,job.get_remaining_avg_time(),self.time_step)
            if job.is_completed():
                self.uncomplete_job.disengage_node(job)
                self.complete_job.append(job)
                job.compute_wait_time(self.time_step)
                self.reward_calculator.update_job_completion(job.id-1,self.time_step,machine.id-1)
            job = next_job
        # print(f'{self.complete_job.length}:{self.uncomplete_job.length}')
        done = True if self.complete_job.length >= self.max_job_num else False
        truncated = True if self.time_step > 950 else False
        while (
            not done and not truncated and not self.is_any_machine_need_to_decision()
        ):  # 没有结束且没有空闲机器，继续
            done, truncated = self.run()
        return done, truncated
    def get_randed_avi_jobs(self,ranked_lists):
        seen_ids = set()
        result = []
        pointers = [0] * len(ranked_lists)  # 每个列表的当前索引

        while len(result) < self.action_dim:
            no_more_candidates = True
            for i, ranked in enumerate(ranked_lists):
                while pointers[i] < len(ranked):
                    job = ranked[pointers[i]]
                    pointers[i] += 1
                    if id(job) not in seen_ids:
                        result.append(job)
                        seen_ids.add(id(job))
                        no_more_candidates = False
                        break  # 当前规则找到一个就换下一个
            if no_more_candidates:
                break  # 所有规则都没新作业了，结束
        return result
    def get_obs_i(self):
        """
        获取machine i 的 obs
        如果可用作业大于5，则用调度规则选取的作业信息作为state
        否则
        """
        ranked_job0 = EDD(self.available_jobs)[:1]
        ranked_job1 = CR(self.available_jobs, self.time_step)[:1]
        ranked_job3 = MS(self.available_jobs, self.time_step)[:1]
        ranked_job2 = SRO(self.available_jobs, self.time_step)[:1]

        update_avi_jobs = self.get_randed_avi_jobs([ranked_job0,ranked_job1,ranked_job2,ranked_job3])
        unique_count = len(set(id(obj) for obj in update_avi_jobs))
        self.count_actions[unique_count - 1] += 1
        obs_i = []
        for i,job,in enumerate(update_avi_jobs):
            code1 = job.get_state_code(self.time_step)
            code2 = [1 if j == i else 0  for j in range(4)]
            code = code1 + code2
            obs_i.append(code)
        # obs_i = [job.get_state_code(self.time_step) for job in update_avi_jobs]
        self.available_jobs = update_avi_jobs

        obs_mask = [False if i < len(obs_i) else True for i in range(self.obs_len)]
        obs_mask[-1] = False
        for i in range(self.obs_len - 1 - len(obs_i)):
            obs_i.append([0 for _ in range(self.obs_dim)])
        all_urgency = self.compute_urgency()
        obs_il = [self.current_machine.id/self.machine_num,np.mean(all_urgency),np.max(all_urgency),np.max(all_urgency)]
        obs_il.extend([job.get_urgency(self.time_step) for job in self.available_jobs])
        for i in range(len(obs_i[0])-len(self.available_jobs)-4):
            obs_il.append(0)
        obs_i.append(obs_il)
        return obs_i, obs_mask

    def step(self, action):
        self.current_machine.load_job(self.available_jobs[action], self.time_step)
        self.current_machine.update_decision_time(self.time_step)
        reward,_ = self.reward_calculator.calculate_system_reward(self.time_step)
        done, truncated = False, False
        cur_job:Job = self.available_jobs[action]
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()
        if not done and not truncated:
            decision_machines = self.get_decsion_machines()
            self.current_machine = decision_machines[0]
            self.available_jobs = self.get_available_jobs()
            obs_i, obs_mask = self.get_obs_i()
            global_state = self.get_global_state()
        else:
            obs_i = [[0 for _ in range(self.obs_dim)] for _ in range(self.obs_len)]
            obs_mask = [True for _ in range(self.obs_len)]
            obs_mask[0] = False
            global_state = [0 for _ in range(6)]
        return obs_i, obs_mask, global_state, reward , done, truncated

    def step_by_sr(self, action):
        self.current_machine.load_job(action, self.time_step)
        self.current_machine.update_decision_time(self.time_step)
        done, truncated = False, False
        if (
            not self.is_any_machine_need_to_decision()
        ):  # 没有机器需要采样动作，直接运行,直到结束，或者有机器需要采样动作
            done, truncated = self.run()
        if not done and not truncated:
            decision_machines = self.get_decsion_machines()
            self.current_machine = decision_machines[0]
            self.available_jobs = self.get_available_jobs()
        return 0, done, truncated

    def is_any_machine_need_to_decision(self):
        for machine in self.machines:
            if machine.is_idle() and self.is_decision_machine(machine):
                return True
        return False
    def compute_UR(self):
        utiliaction_rates = [
            agent.get_utilization_rate(self.time_step) for agent in self.machines
        ]
        return utiliaction_rates

    def compute_idle_time(self):
        idle_times = [agent.get_idle_time(self.time_step) for agent in self.machines]
        return idle_times

    def compute_slack_time(self):
        slack_time = []
        job = self.uncomplete_job.head
        while job:
            slack_time.append(job.get_slack_time(self.time_step))
            job = job.next
        return slack_time
    def compute_urgency(self):
        urgency = []
        job = self.uncomplete_job.head
        while job:
            urgency.append(job.get_urgency(self.time_step))
            job = job.next
        return urgency