import d4rl
import gym
import argparse
import pickle
import numpy as np

def normalize(env, score):
    return env.get_normalized_score(score)*100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper-medium-replay-v2')
    parser.add_argument('--model', type=str, default='TrajectoryTransformer')
    parser.add_argument('-t', '--trajectories', type=str, default='./replays/TT/hopper-medium-replay-v2.pkl')
    args = parser.parse_args()
    env = gym.make(args.env)
    print("=" * 20)
    print(f"Environment: {args.env}")
    print(f"Ref Min Score: {env.ref_min_score}")
    print(f"Ref Max Score: {env.ref_max_score}")
    print("=" * 20)

    with open(args.trajectories, 'rb') as f:
        data = pickle.load(f)

    scores = []
    for i, d in enumerate(data):
        try:
            total_reward, trajectory = d
        except:
            total_reward, _, trajectory = d
        # print(normalize(env, float(reward)))
        scores.append(normalize(env, total_reward))
    np.array(scores)

    # Get the mean and std of the scores
    mean = np.mean(scores)
    std = np.std(scores)

    print("Model: {}".format(args.model))
    print("Mean: {}".format(mean))
    print("Std: {}".format(std))