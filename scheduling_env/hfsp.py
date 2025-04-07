import os
from typing import SupportsFloat, Any, Union, Optional

import gymnasium as gym
import matplotlib
from matplotlib.patches import Rectangle

# matplotlib.use('TkAgg', force=True)
# print("Switched to:", matplotlib.get_backend())
import matplotlib.pyplot as plt

import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
import pandas as pd

# from line_profiler_pycharm import profile

INF = 1000


def fetch_jobs_from_file(path: str):
    with open(path, "r") as f:
        job_num, machine_num, avg_machine_job = map(int, f.readline().split())
        job_lists: dict[int, dict] = dict()
        max_op_num = 0
        for job_id, line_str in enumerate(f):
            line = list(map(int, line_str.split()))
            i, r = 1, 0
            job: dict[int, dict[int, int]] = dict()  # 工序列表
            while i < len(line):
                # s = f'工序{r} '
                op: dict[int, int] = {}  # 单个工序
                for j in range(line[i]):
                    # s +=f'机器{line[i+1+j*2]} 耗时{line[i+1+j*2+1]} || '
                    op[line[i + 1 + j * 2] - 1] = line[i + 1 + j * 2 + 1]
                job[r] = op
                r += 1
                i += 1 + line[i] * 2
            job_lists[job_id] = job
            max_op_num = max(max_op_num, len(job))

    job_op_num = np.zeros((job_num,))
    job_machine_op_matrix = (
        np.zeros((job_num, machine_num, max_op_num + 1), dtype=int) + INF
    )
    for job_id, job in job_lists.items():
        job_op_num[job_id] = len(job)
        for op_id, op in job.items():
            for machine_id, during_time in op.items():
                job_machine_op_matrix[job_id, machine_id, op_id] = during_time

    return job_machine_op_matrix, job_op_num


class HFSPEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        self.machine_op_dict = None
        self.IDlE = None
        self.current_machine2job_matrix = None
        self.job_op_num = None
        self._rows = None
        self.current_time = None
        self.current_job_state = None
        self.job_machine_op_matrix = None
        self.action_dim = 21
        self.action_space = gym.spaces.MultiDiscrete([self.action_dim] * 15)
        self.observation_space = gym.spaces.Box(
            np.array([0] * 20 + [-INF] * 20, dtype=int),
            np.array([20] * 20 + [INF] * 20, dtype=int),
            dtype=np.int32,
        )
        self.fig = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        np.random.seed(seed)

        train_data_path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/data/"
            + "train_data/Mk10.fjs"
        )
        self.job_machine_op_matrix, self.job_op_num = fetch_jobs_from_file(
            train_data_path
        )
        self.machine_op_dict = dict()
        self.IDlE = self.action_dim - 1
        self.job_dim = self.job_machine_op_matrix.shape[0]
        self.action_dim = self.job_machine_op_matrix.shape[1]

        for mi in range(self.action_dim):
            self.machine_op_dict[mi]: dict[int, set] = dict()
            for op in range(self.job_machine_op_matrix.shape[2]):
                self.machine_op_dict[mi][op] = set(
                    [
                        job
                        for job, v in enumerate(self.job_machine_op_matrix[:, mi, op])
                        if v < INF
                    ]
                    + [self.IDlE]
                )

        self.current_job_state = np.zeros(
            (2,) + (self.job_dim,), dtype=int
        )  # 0-op id, 1-makespan
        self.current_machine2job_matrix = np.zeros(
            self.job_machine_op_matrix.shape[:2]
        )  # all idle
        self.current_time = 0
        self._rows = np.arange(0, self.job_machine_op_matrix.shape[0])
        return self.get_state(), {}

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        if self.fig is None:
            _, self.fig = plt.subplots()

        image = np.zeros(
            (
                self.job_machine_op_matrix.shape[0],
                self.job_machine_op_matrix.shape[2],
                3,
            )
        )
        for i, (j, k) in enumerate(zip(*self.current_job_state)):
            machine = np.where(self.current_machine2job_matrix[i, :])[0]
            image[i, j] = [min(max(k, 0) / 10, 1), 0, 0]
            image[i, :j] = [0.0, 0.3, 0.1]
            if len(machine) == 0:
                image[i, j, :] = [0, 0, 1]
        # plt.title('Interesting Graph')
        image = image / image.max()

        plt.imshow(image)
        plt.draw()
        plt.pause(0.05)

    def close(self):
        if self.fig is not None:
            self.fig.close()

    # @profile
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert len(action) == self.action_space.shape[0], "action dim error"
        reward = 0
        action_copy = action.copy()

        idle_actions_tag = action >= self.IDlE

        machine_running = self.current_machine2job_matrix.sum(axis=0) > 0
        idle_actions_tag = idle_actions_tag + machine_running

        op_running = [
            (False if a >= self.IDlE else (self.current_job_state[1, a] > 0))
            for a in action
        ]
        idle_actions_tag = idle_actions_tag | machine_running | op_running

        action[idle_actions_tag] = self.action_dim - 1

        idle_machine_tag = (self.current_machine2job_matrix > 0).sum(axis=0) == 0

        #  incompatible actions   njob x nmach  nmach x 1
        available_job_set = set(np.where(self.current_job_state[1, :] <= 0)[0])
        if len(available_job_set) > 0:
            available_job_list = np.asarray(list(available_job_set))
            for mi, job in enumerate(action):
                if not idle_machine_tag[mi]:
                    action[mi] = self.IDlE
                    continue
                if (
                    self.job_machine_op_matrix[job, mi, self.current_job_state[0, job]]
                    < INF
                    and self.current_job_state[1, job] <= 0
                ):  # compatible
                    continue
                js = set(
                    available_job_list[
                        np.where(
                            self.job_machine_op_matrix[
                                available_job_list,
                                mi,
                                self.current_job_state[0, available_job_list],
                            ]
                            < INF
                        )[0]
                    ]
                )
                js = js & available_job_set
                js.add(self.IDlE)
                action[mi] = list(js)[action[mi] % len(js)]
        else:
            action[:] = [self.IDlE] * self.action_dim

        op_repeat = [(v != self.IDlE and v in action[:i]) for i, v in enumerate(action)]
        action[op_repeat] = self.IDlE

        # assign op to machine
        available_actions_tag = action < self.IDlE
        available_actions = action[available_actions_tag]
        makespan = self.job_machine_op_matrix[
            available_actions,
            available_actions_tag,
            self.current_job_state[0, available_actions],
        ]
        if any(makespan >= INF):
            raise ValueError("assign a incompatible job", available_actions)

        if any(self.current_job_state[1, available_actions] > 0):
            raise ValueError("assign a running job to machine")

        if any((self.current_machine2job_matrix > 0).sum(axis=0) > 1):
            raise ValueError("assign job to a working machine")

        self.current_job_state[0, available_actions] += 1
        self.current_job_state[1, available_actions] = makespan
        self.current_machine2job_matrix[available_actions, available_actions_tag] = (
            available_actions
        )
        self.current_time += 1
        self.current_job_state[1, :] -= 1
        self.current_machine2job_matrix[self.current_job_state[1, :] <= 0, :] *= 0

        # op is completed ?
        if all(
            self.current_job_state[1, :]
            <= 0 & np.greater_equal(self.current_job_state[0, :], self.job_op_num)
        ):
            terminated = True
            reward += 1000 / self.current_time
            print(self.current_time)
        else:
            terminated = False

        truncated = self.current_time >= 400
        return self.get_state(), reward, terminated, truncated, {}

    def get_state(self):
        return self.current_job_state.ravel()


if __name__ == "__main__":
    gym.envs.register(
        id="HFSP-v0",
        entry_point="hfsp:HFSPEnv",
        # vector_entry_point="",
        max_episode_steps=400,
        reward_threshold=197.0,
    )
    env: HFSPEnv = gym.make("HFSP-v0")
    for i in range(100):
        env.reset()
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            state, reward, terminated, truncated, infos = env.step(action)
            terminated = terminated or truncated
    # train_data_path = os.path.dirname(os.path.abspath(__file__)) + '/data/' + 'train_data/Mk10.fjs'
    # fetch_jobs_from_file(train_data_path)
