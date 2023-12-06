from absl import app, flags
from ml_collections import config_flags
import gym
from decision_transformer import wrappers
import os
from decision_transformer.dataset_utils import D4RLDataset
from tqdm import tqdm
import numpy as np
import pickle
import jax 
import jax.numpy as jnp
import torch

flags.DEFINE_string('env_name', 'hopper-medium-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './logs/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('use_reward_model', False, 'Use reward model for relabeling reward.')
flags.DEFINE_string('model_type', 'MLP', 'type of reward model.')
flags.DEFINE_string('ckpt_dir',
                    './logs/pref_reward_model/hopper-medium-expert-v2/PrefTransformer/attn_seq25/s42',
                    'ckpt path for reward model.')
flags.DEFINE_string('comment',
                    'base',
                    'comment for distinguishing experiments.')
flags.DEFINE_integer('seq_len', 25, 'sequence length for relabeling reward in Transformer.')
flags.DEFINE_bool('use_diff', False, 'boolean whether use difference in sequence for reward relabeling.')
flags.DEFINE_string('label_mode', 'last', 'mode for relabeling reward with tranformer.')
flags.DEFINE_boolean('with_attn_weights', True, 'boolean whther attn layer is used as output layert')
config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

FLAGS = flags.FLAGS
@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)

def get_preferences(reward_model, states, actions, timesteps, attn_mask, with_attn_weights=True, label_mode='last'):
    '''
    Get human feedback embeddings from pretrained Preference Transformer.

    Parameters:
    - states (torch.Tensor): The states in the trajectory, shape (batch_size, seq_length, state_dim).
    - actions (torch.Tensor): The actions in the trajectory, shape (batch_size, seq_length, act_dim).
    - rewards (torch.Tensor, optional): The existing rewards, shape (batch_size, seq_length, 1).
    - timesteps (torch.Tensor): The timesteps in the trajectory, shape (batch_size, seq_length).

    Returns:
    - torch.Tensor: the human feedback tensor (shape (batch_size, seq_length, human_feedback_dim))
    '''
    #  turn input as numpy arrays
    states = states.cpu().numpy()
    actions = actions.cpu().numpy()
    timesteps = timesteps.cpu().numpy()
    attn_mask = attn_mask.cpu().numpy()

    input = dict(
        observations=states,
        actions=actions,
        timestep=timesteps,
        attn_mask=attn_mask,
    )
    attn_weights = []
    # turn input as jax tree
    jax_input = batch_to_jax(input)
    # if FLAGS.with_attn_weights:
    if with_attn_weights:
        new_reward, attn_weight = reward_model.get_reward(jax_input)
        # new_reward: shape (batch, seq_len, human_pref_dim(1 as deafult))
        # attn_weight: shape (batch, num_head, seq_len, seq_len)
        attn_weights.append(np.array(attn_weight[:,0,-1,:]))
    else:
        new_reward, _ = reward_model.get_reward(jax_input)

    # if FLAGS.label_mode == "mean":
    if label_mode == "mean":
        new_reward = jnp.sum(new_reward, axis=1) / jnp.sum(attn_mask, axis=1)
        new_reward = new_reward.reshape(-1, 1)
    # elif FLAGS.label_mode == "last":
    elif label_mode == "last":
        new_reward = new_reward[:, -1].reshape(-1, 1)
    # turn reward from jax array to torch tensor
    new_reward = torch.Tensor(np.asarray(list(new_reward))).to("cuda:0")
    attn_weight = torch.Tensor(np.asarray(list(attn_weight))).to("cuda:0")

    return new_reward, attn_weight



def main(_):
    env = gym.make(FLAGS.env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)
    # create dataset
    dataset = D4RLDataset(env)

    # preprocessing dataset
    _obs = []
    _act = []
    traj_len = 0
    trajectories = []
    trj_mapper = []
    observation_dim = dataset.observations.shape[-1]
    action_dim = dataset.actions.shape[-1]
    traj_counter = 0
    for i in tqdm(range(len(dataset.observations)), desc="split"):
        # trajs[-1].append((dataset.observations[i], dataset.actions[i]))
        _obs.append(dataset.observations[i])
        _act.append(dataset.actions[i])
        traj_len += 1
        if dataset.dones_float[i] == 1.0 and i + 1 < len(dataset.observations):
            _obs, _act = np.asarray(_obs), np.asarray(_act)
            trajectories.append((_obs, _act))
            for seg_idx in range(traj_len):
                trj_mapper.append((traj_counter, seg_idx))
            traj_counter += 1
            _obs = []
            _act = []
            traj_len = 0
    
    data_size = dataset.rewards.shape[0]
    interval = int(data_size / FLAGS.batch_size) + 1
    new_r = np.zeros_like(dataset.rewards)
    pts = []
    attn_weights = []
    # load model
    if os.path.exists(os.path.join(FLAGS.ckpt_dir, "best_model.pkl")):
        model_path = os.path.join(FLAGS.ckpt_dir, "best_model.pkl")
    else:
        model_path = os.path.join(FLAGS.ckpt_dir, "model.pkl")
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt['reward_model']

    # TODO: make a a jax function
    # truncate or pad the trajectory into seq_len
    # for i in range(interval, desc="relabel reward"):
    for i in tqdm(range(10), desc="relabel reward"): # for debug
        start_pt = i * FLAGS.batch_size
        end_pt = min((i + 1) * FLAGS.batch_size, data_size)

        _input_obs, _input_act, _input_timestep, _input_attn_mask, _input_pt = [], [], [], [], []
        for pt in range(start_pt, end_pt):
            _trj_idx, _seg_idx = trj_mapper[pt]
            if _seg_idx < FLAGS.seq_len - 1:
                __input_obs = np.concatenate([np.zeros((FLAGS.seq_len - 1 - _seg_idx, observation_dim)), trajectories[_trj_idx][0][:_seg_idx + 1, :]], axis=0)
                __input_act = np.concatenate([np.zeros((FLAGS.seq_len - 1 - _seg_idx, action_dim)), trajectories[_trj_idx][1][:_seg_idx + 1, :]], axis=0)
                __input_timestep = np.concatenate([np.zeros(FLAGS.seq_len - 1 - _seg_idx, dtype=np.int32), np.arange(1, _seg_idx + 2, dtype=np.int32)], axis=0)
                __input_attn_mask = np.concatenate([np.zeros(FLAGS.seq_len - 1 - _seg_idx, dtype=np.int32), np.ones(_seg_idx + 1, dtype=np.float32)], axis=0)
                __input_pt = np.concatenate([np.zeros(FLAGS.seq_len - 1 - _seg_idx), np.arange(pt - _seg_idx , pt + 1)], axis=0)
            else:
                __input_obs = trajectories[_trj_idx][0][_seg_idx - FLAGS.seq_len + 1:_seg_idx + 1, :]
                __input_act = trajectories[_trj_idx][1][_seg_idx - FLAGS.seq_len + 1:_seg_idx + 1, :]
                __input_timestep = np.arange(1, FLAGS.seq_len + 1, dtype=np.int32)
                __input_attn_mask = np.ones((FLAGS.seq_len), dtype=np.float32)
                __input_pt = np.arange(pt - FLAGS.seq_len + 1, pt + 1)

            _input_obs.append(__input_obs)
            _input_act.append(__input_act)
            _input_timestep.append(__input_timestep)
            _input_attn_mask.append(__input_attn_mask)
            _input_pt.append(__input_pt)
        # shape : [batch, seq_len]
        _input_obs = np.asarray(_input_obs)
        _input_act = np.asarray(_input_act)
        _input_timestep = np.asarray(_input_timestep)
        _input_attn_mask = np.asarray(_input_attn_mask)
        _input_pt = np.asarray(_input_pt)
        # to torch tensor
        _input_obs = torch.Tensor(_input_obs) #shape :(batch, seq_len, obs_dim)
        _input_act = torch.Tensor(_input_act) #shape :(batch, seq_len, action_space)
        _input_timestep = torch.Tensor(_input_timestep).type(torch.int32) #shape :(batch, seq_len)
        _input_attn_mask = torch.Tensor(_input_attn_mask) #shape :(batch, seq_len)
        
        reward, attn_weights = get_preferences(reward_model, _input_obs, _input_act, _input_timestep, _input_attn_mask)
        print(type(reward), reward.shape)
        print(type(attn_weights), attn_weights.shape)
        break

if __name__ == "__main__":
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    app.run(main)
