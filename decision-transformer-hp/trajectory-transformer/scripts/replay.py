import os
import json
import pdb
from os.path import join
import pickle
import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    beam_plan,
    make_prefix,
    extract_actions,
    update_context,
)

class Parser(utils.Parser):
    dataset: str = 'halfcheetah-medium-expert-v2'
    config: str = 'config.offline'

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')

#######################
####### models ########
#######################

dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
        'data_config.pkl')

gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
        epoch=args.gpt_epoch, device=args.device)

#######################
####### dataset #######
#######################

env, seed = datasets.load_environment(args.dataset)
timer = utils.timer.Timer()

discretizer = dataset.discretizer
discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

value_fn = lambda x: discretizer.value_fn(x, args.percentile)
preprocess_fn = datasets.get_preprocess_fn(env.name)

#######################
###### main loop ######
#######################

T = env.max_episode_steps

if not os.path.exists('{}'.format('replays')):
    os.mkdir('{}'.format('replays'))

trajectories = []
for i in range(5):

    observation = env.reset()
    total_reward = 0

    ## observations for rendering
    rollout = [observation.copy()]
    states = []
    actions = []
    rewards = []

    ## previous (tokenized) transitions for conditioning transformer
    context = []

    for t in range(T):

        observation = preprocess_fn(observation)


        if t == 0:
            states.append(observation.copy())

        if t % args.plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            prefix = make_prefix(discretizer, context, observation, args.prefix_context)

            ## sample sequence from model beginning with `prefix`
            sequence = beam_plan(
                gpt, value_fn, prefix,
                args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                discount, args.max_context_transitions, verbose=args.verbose,
                k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
            )

        else:
            sequence = sequence[1:]

        ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
        sequence_recon = discretizer.reconstruct(sequence)

        ## [ action_dim ] index into sampled trajectory to grab first action
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)
        
        actions.append(action.copy())

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)
        states.append(next_observation.copy())
        rewards.append(reward)

        ## update return
        total_reward += reward
        score = env.get_normalized_score(total_reward)


        ## update rollout observations and context transitions
        rollout.append(next_observation.copy())
        context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions)

        print(
            f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
            f'time: {timer():.2f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
        )

        if terminal: 
            break

        observation = next_observation

    ## save result as a json file
    json_path = join(args.savepath, 'rollout.json')
    json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal, 'gpt_epoch': gpt_epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)


    ## save rollout
    trajectories.append([total_reward, {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'seed': seed,
    }])

    with open('{}/{}.pkl'.format('replays', args.dataset, i), 'wb') as f:
        pickle.dump(trajectories, f)