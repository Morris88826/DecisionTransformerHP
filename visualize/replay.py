import torch
import gym
import time
import d4rl
import pickle
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-expert-v2")
    parser.add_argument("-t", "--trajectory", type=str, default="./replays/DT+PT/hopper-medium-expert-v2/10/best_traj_36000.pkl")
    args = parser.parse_args()

    print("Loading environment: ", args.dataset)
    print("Loading trajectory: ", args.trajectory)

    env = gym.make(args.dataset)

    with open(args.trajectory, "rb") as f:
        data = pickle.load(f)

    max_reward = -10000
    best_trajectory = None
    for d in data:
        try:
            total_reward, trajectory = d
        except:
            total_reward, _, trajectory = d

        if total_reward > max_reward:
            max_reward = total_reward
            best_trajectory = trajectory

    while True:
        seed = best_trajectory["seed"]
        try:
            states = best_trajectory["states"].cpu().numpy()
            actions = best_trajectory["actions"].cpu().numpy()
            rewards = best_trajectory["rewards"].cpu().numpy()
        except:
            states = best_trajectory["states"]
            actions = best_trajectory["actions"]
            rewards = best_trajectory["rewards"]

        # need to set the seed before reset
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset()

        for state, action in zip(states[:-1], actions):
            env.unwrapped.state = state
            next_state, reward, done, _  = env.step(action)

            env.render()
            if done:
                break


