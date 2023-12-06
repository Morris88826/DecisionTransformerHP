import mujoco_py
import gym
import numpy as np
import d4rl
import argparse
import pickle
import trajectory.utils as utils
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-expert-v2")
    parser.add_argument("-t", "--trajectory", type=str, default="./replays/DT+PT/hopper-medium-expert-v2/10/best_traj_36000.pkl")
    parser.add_argument("--renderer", default="Renderer")
    args = parser.parse_args()

    print("Loading environment: ", args.dataset)
    print("Loading trajectory: ", args.trajectory)


    with open(args.trajectory, "rb") as f:
        data = pickle.load(f)

    env = gym.make(args.dataset)


    renderer = utils.make_renderer(args)

    out_path = "demo"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, args.dataset)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for i,d in enumerate(data):
        try:
            total_reward, trajectory = d
        except:
            total_reward, _, trajectory = d

        seed = trajectory["seed"]
        try:
            states = trajectory["states"].cpu().numpy()
            actions = trajectory["actions"].cpu().numpy()
            rewards = trajectory["rewards"].cpu().numpy()
        except:
            states = trajectory["states"]
            actions = trajectory["actions"]
            rewards = trajectory["rewards"]

        print("Total reward: ", total_reward)
        print("Seed: ", seed)
        print("States: ", len(states))
        print("Actions: ", len(actions))
        print("Rewards: ", len(rewards))
        
        # need to set the seed before reset
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset()

        renderer.render_rollout(os.path.join(out_path, f'sample_{i}.mp4'), states, fps=80)