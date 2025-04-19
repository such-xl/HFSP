from enum import Enum
import numpy as np
from .utils import Node, DoublyLinkList


class JobStatus(Enum):
    COMPLETED = 0
    IDLE = 1
    RUNNING = 2


class Job(Node):
    def __init__(
        self, id: int, type: int, process_num: int, process_list: list, insert_time: int, due_time: int
    ) -> None:
        super().__init__(None)
        self._id = id  # job序号,从1开始
        self._process_num = process_num  # job工序数
        self._type = type
        self._process_list = (
            process_list  # job工序列表[{机器1:加工时间1,机器2:加工时间2},...{}]
        )
        self._due_time = due_time  # job截止时间
        self._progress = 1  # 加工进度 代表第progess道工序待加工，0 代表加工完成
        self._status = JobStatus.IDLE
        self._machine = None  # 正在加工该job的机器id，0表示目前没有被加工
        self._t_process = 0  # 当前工序需被加工的时间
        self._t_processed = 0  # 当前工序已经被加工时间
        self._insert_time = insert_time  # 作业插入时间
        self._process_time = 0  # 作业总加工时间
        self._completed_time = 0  # 作业完成时间
        self._wait_time = 0  # 作业等待时间
        self._tard_time = 0

    def get_state_code(self,time_step):
        """
        
        """
        remaining_time = self.get_remaining_avg_time()
        progress_ratio = self._progress / self._process_num
        avg_speed = self._process_time / (time_step - self._insert_time + 1e-6)
        remaining_due_ratio = max(self._due_time - time_step, 0) / (self._due_time - self._insert_time + 1e-6)
        tightness = remaining_time / (self._due_time - time_step + 1e-6)
        tightness = min(tightness, 1.0)  # 可选的clipping，防止过大

        return [
            progress_ratio,
            avg_speed,
            remaining_due_ratio,
            tightness
        ]
    def get_slack_time(self,time_step):
        return max(self._due_time - time_step-self.get_remaining_avg_time(),0)
    def get_urgency(self,time_step):
        urgency = 0
        slack_time = self.get_slack_time(time_step)
        if slack_time <= 0:
            urgency = 1
        else:
            urgency = 1 / slack_time
        return urgency
    def get_t_process(self, machine_id):
        """
        获取当前工序在机器machine上的加工时间

        """
        if self.is_completed():
            raise ValueError("job is completed")
        return self._process_list[self._progress - 1][machine_id]

    def match_machine(self, machine_id) -> bool:
        """
        判断当前工序是否可以被机器machine_id加工
        """
        return machine_id in self._process_list[self._progress - 1]

    def load_to_machine(self, machine, time_step):
        """将job装载至machine"""
        if self._status != JobStatus.IDLE:
            raise ValueError("job is not idle")
        if self._status == JobStatus.COMPLETED:
            raise ValueError("job is completed")
        if self._machine:
            raise ValueError("job has machine")
        self._t_process = self.get_t_process(machine.id)
        self._machine = machine
        self._t_processed = 0
        self._status = JobStatus.RUNNING
        self._rest = time_step

    def unload_machine(self):
        """将job从machine卸载"""
        if self._status != JobStatus.RUNNING:
            raise ValueError("job is not running")
        if not self._machine:
            raise ValueError("job has no machine")

        # print(f'j机器{self.machine.id} unload job {self.id}')
        # self._record[-1][-1] += self._t_processed
        self._t_process = 0
        self._t_processed = 0
        self._progress += 1
        self._status = (
            JobStatus.COMPLETED
            if self._progress == self._process_num + 1
            else JobStatus.IDLE
        )
        if self._status != JobStatus.COMPLETED:
                self._machine = None

    def is_completed(self):
        """判断是否所有工序都完成"""
        return self._status == JobStatus.COMPLETED

    def is_wating_for_machine(self):
        """判断是否等待机器加工"""
        return self._status == JobStatus.IDLE

    def is_on_processing(self):
        """判断是否正在加工"""
        return self._status == JobStatus.RUNNING

    def compute_wait_time(self, time_step):

        self._wait_time = (time_step - self._insert_time) - self._process_time
        self._tard_time = max(0, time_step - self._due_time)

    def run(self, min_run_timestep, time_step):
        """执行min_run_timestep 时序"""
        self._t_processed += min_run_timestep
        self._process_time += min_run_timestep
        if min_run_timestep <= 0:
            raise ValueError("min_run_timestep must be greater than 0")

        if self._t_processed > self._t_process:
            raise ValueError(
                "加工时间超过了工序时间",
                self._t_processed,
                self._t_process,
                min_run_timestep,
                self._status,
                self._machine,
            )

        if self._t_processed == self._t_process:  # 当前工序加工完成
            self.unload_machine()
            self._est = time_step + min_run_timestep

    def current_progress_remaining_time(self):
        """获取当前工序剩余加工时间"""
        if self._status == JobStatus.COMPLETED:
            raise ValueError("job is completed")
        reminder = 0
        if self._status == JobStatus.IDLE:
            times = self._process_list[self._progress - 1].values()
            reminder = sum(times) / len(times)
            # reminder = min(self._process_list[self._progress-1].values())
        elif self._status == JobStatus.RUNNING:
            reminder = self._t_process - self._t_processed
        if reminder <= 0:
            raise ValueError("reminder is negative or zero")
        return reminder

    def get_remaining_avg_time(self):
        """获取平均剩余加工时间"""
        if self._status == JobStatus.COMPLETED:
            return 0
        reminder = 0
        if self._status == JobStatus.IDLE:
            reminder = sum(self._process_list[self._progress - 1].values()) / len(
                self._process_list[self._progress - 1].values()
            )
        else:
            reminder = self._t_process - self._t_processed

        reminder_progress = self._process_list[self._progress :]
        for p in reminder_progress:
            reminder += sum(p.values()) / len(p.values())
        if reminder <= 0:
            raise ValueError("reminder is negative or zero")
        return reminder

    @property
    def id(self):
        return self._id

    @property
    def type(self):
        return self._type

    @property
    def process_num(self):
        return self._process_num

    @property
    def process_list(self):
        return self._process_list

    @property
    def progress(self):
        return self._progress

    @property
    def status(self):
        return self._status

    @property
    def machine(self):
        return self._machine

    @property
    def t_process(self):
        return self._t_process

    @property
    def t_processed(self):
        return self._t_processed

    @property
    def insert_time(self):
        return self._insert_time

    @property
    def process_time(self):
        return self._process_time

    @property
    def completed_time(self):
        return self._completed_time

    @property
    def wait_time(self):
        return self._wait_time
    @property
    def due_time(self):
        return self._due_time
    @property
    def tard_time(self):
        return self._tard_time
    @due_time.setter
    def due_time(self, due_time):
        self._due_time = due_time

class JobList(DoublyLinkList):
    def __init__(self) -> None:
        super().__init__()


def fetch_job_info(path: str):
    """
    从文件中获取作业信息
    return: 机器数，作业信息
    """
    machine_num: int = 0
    with open(path, "r") as f:

        _, machine_num = map(int, f.readline().split()[0:-1])
        job_info = []
        for type, line_str in enumerate(f, start=1):

            line = list(map(int, line_str.split()))

            i, r = 1, 1

            procs: list[dict[int, int]] = []  # 工序列表
            while i < len(line):
                # s = f'工序{r} '
                proc: dict[int, int] = {}  # 单个工序
                for j in range(line[i]):
                    # s +=f'机器{line[i+1+j*2]} 耗时{line[i+1+j*2+1]} || '
                    proc[line[i + 1 + j * 2]] = line[i + 1 + j * 2 + 1]

                procs.append(proc)
                r += 1
                i += 1 + line[i] * 2
            job_info.append({"type": type, "process_num": r - 1, "process_list": procs})
    return machine_num, job_info
