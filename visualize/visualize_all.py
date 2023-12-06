import mujoco_py
import gym
import numpy as np
import d4rl
import argparse
import pickle
import trajectory.utils as utils
import os
import glob

def visualize(dataset, model, name, data):

    print("Replaying: {} for model {}, {}~".format(dataset, model, name))

    renderer = utils.make_renderer(dataset)

    out_path = "results"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, model)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, dataset)
    if not os.path.exists(out_path):
        os.mkdir(out_path)


    env = gym.make(dataset)

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

        try:
            renderer.render_rollout(os.path.join(out_path, f'{name}_sample_{i}_reward_{total_reward}.mp4'), states, fps=80)
        except:
            print("Rendering failed for {}_sample_{}.mp4".format(name, i))
            continue

if __name__ == "__main__":
    
    models = ['IQL', 'DT', 'TT', 'DT+PT']

    replay_dir = "./replays"

    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for model in models:
        # if model == "IQL":
        #     replays = glob.glob(os.path.join(replay_dir, "IQL", "*", "*.pkl"))

        #     for replay_path in replays:
        #         info = replay_path.split("/")
        #         model_name = info[-3]
        #         dataset = info[-2]
        #         name = info[-1].split(".")[0]
        #         with open(replay_path, "rb") as f:
        #             data = pickle.load(f)
        #         visualize(dataset, model_name, name, data)

        # if model == "DT":
        #     replays = glob.glob(os.path.join(replay_dir, "DT", "*", "*", "*.pkl"))

        #     for replay_path in replays:
        #         info = replay_path.split("/")
        #         model_name = info[-4]
        #         dataset = info[-3]
        #         iteration = info[-2]
        #         name = info[-1].split(".")[0]
        #         name += "_iter{}_".format(iteration)
        #         with open(replay_path, "rb") as f:
        #             data = pickle.load(f)
        #         visualize(dataset, model_name, name, data)

        if model == "TT":
            replays = glob.glob(os.path.join(replay_dir, "TT", "*.pkl"))
            
            for replay_path in replays:
                info = replay_path.split("/")
                model_name = info[-2]
                dataset = info[-1].split(".")[0]
                name = dataset
                with open(replay_path, "rb") as f:
                    data = pickle.load(f)
                visualize(dataset, model_name, name, data)

        # if model == "DT+PT":
        #     replays = glob.glob(os.path.join(replay_dir, "DT+PT", "*", "*", "*.pkl"))
            
        #     for replay_path in replays:
        #         info = replay_path.split("/")
        #         model_name = info[-4]
        #         dataset = info[-3]
        #         iteration = info[-2]
        #         name = info[-1].split(".")[0]
        #         name += "_iter{}_".format(iteration)
        #         with open(replay_path, "rb") as f:
        #             data = pickle.load(f)
        #         visualize(dataset, model_name, name, data)
