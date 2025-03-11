import math
from .job import Job,JobList
import numpy as np
from .machine import Machine,MachineList
class TrainingEnv():
  def _init__(self,action_dim,reward_type,max_machine_num,max_job_num):
    self._action_space = (0,action_dim-1)
    self._action_dim = action_dim
    self._max_machine_num = max_machine_num
    self._max_job_num = max_job_num
    self._completed_jobs = JobList()
    self._uncompleted_jobs = JobList()
    self._busy_machines = MachineList(0)
    self._faulty_machines = MachineList(0)
    self._idle_machines = None
    self._machine_list = []
    self._machines = None
    self._draw_data = None
    self._time_step = 0
    self._reward_type = reward_type
    self._decision_machines = None
    self._job_list = None

  def get_jobs_from_file(self, jobs_path:str):
      self._max_machine_num = self._uncompleted_jobs.fetch_jobs_from_file(jobs_path)
      self._max_job_num = self._uncompleted_jobs.length
      self._machines = MachineList(self._max_machine_num)
      self._idle_machines = self._machines
      machines_list = self._machines.head
      while machines_list:
          self._machine_list.append(machines_list)
          machines_list = machines_list.next
    
  def reset(self):
    self._time_step = 0
    self.get_jobs_from_file()