import json
from params import PARAMS
from scheduling_env.training_env import TrainingEnv
from scheduling_env.model import PPO


def train_mappo(env, ppo, num_episodes):
    record = {
        "makespan": [],
        "utiliaction": [],
        "slack_time": [],
        "loss": [],
        "reward": [],
    }

    for episode in range(num_episodes):
        actions = [0, 0, 0, 0]
        locals_state = env.reset(1)
        global_state = env.get_global_state()
        done, truncated = False, False
        while not done and not truncated:
            action = ppo.take_action(locals_state)
            actions[action] += 1
            next_locals_state, reward, done, truncated = env.step(action)
            next_global_state = env.get_global_state()
            ppo.store_transition(
                locals_state,
                global_state,
                action,
                reward,
                next_locals_state,
                next_global_state,
                done,
            )
            global_state = next_global_state
            locals_state = next_locals_state
        actor_loss, loss_U, loss_trad = ppo.update()
        record["makespan"].append(env.time_step)
        record["utiliaction"].append(env.compute_machine_utiliaction())
        record["slack_time"].append(env.compute_slack_time())
        record["loss"].append(actor_loss)
        record["reward"].append(reward)
        print(
            f"Episode {episode + 1}/{num_episodes}:, Actor Loss {actor_loss:.4f}, loss_U {loss_U:.4f}, loss_tard {loss_trad:.4f},slack_time {env.compute_slack_time():.4f} make_span {env.time_step},actions:{actions}"
        )
    with open(
        f"HFSD/record/record_{env.machine_num}_{env.E_ave}_{env.new_insert}_RL.json",
        "w",
    ) as f:  # machine E_ave new_insert
        json.dump(record, f)
    ppo.save_model(f"HFSD/models/ppo_model_plt_RL.pth")


def step_by_sr(env, num_episodes, action, name):
    record = {
        "makespan": [],
        "utiliaction": [],
        "slack_time": [],
        "utiliaction_std": [],
        "reward": [],
    }
    for episode in range(num_episodes):
        actions = [0, 0, 0, 0, 0]
        _ = env.reset(1)
        done, truncated = False, False
        while not done and not truncated:
            actions[action] += 1
            reward, done, truncated = env.sr(action)

        record["makespan"].append(env.time_step)
        record["utiliaction"].append(reward[0])
        record["slack_time"].append(reward[1])
        record["utiliaction_std"].append(env.U_R())
        record["reward"].append(reward)
        print(
            f"Episode {episode + 1}/{num_episodes}: makespan {env.time_step}, slack_time {reward[1]:.4f}  actions {actions}"
        )

    with open(f"HFSD/records/record_{name}.json", "w") as f:
        json.dump(record, f)


env = TrainingEnv(
    action_dim=PARAMS["action_dim"],
    machine_num=PARAMS["machine_num"],
    E_ave=PARAMS["E_ave"],
    new_insert=PARAMS["new_insert"],
)
ppo = PPO(
    local_state_dim=PARAMS["local_state_dim"],
    local_state_len=PARAMS["local_state_len"],
    global_state_dim=PARAMS["global_state_dim"],
    global_state_len=PARAMS["global_state_len"],
    act_dim=PARAMS["action_dim"],
    a_lr=PARAMS["actor_lr"],
    c_lr=PARAMS["critic_lr"],
    gamma=PARAMS["gamma"],
    lmbda=PARAMS["lmbda"],
    epochs=PARAMS["epochs"],
    eps=PARAMS["eps"],
    device=PARAMS["device"],
    weights=PARAMS["weights"],
    batch_size=PARAMS["batch_size"],
)

train_mappo(
    env=env,
    ppo=ppo,
    num_episodes=PARAMS["num_episodes"],
)


sr = ["SPT", "LPT", "LRPT", "Random", "FIFO"]
action = 4
# step_by_sr(
#         env=env,
#         num_episodes=PARAMS["num_episodes"],
#         action=action,
#         name = sr[action]
#     )
