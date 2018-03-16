import time
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from envs import create_sc2_minigame_env
from envs import GameInterfaceHandler
from model import FullyConv
from summary import Summary


def monitor_fn(rank, args, shared_model, global_episode_counter, summary_queue):
    torch.manual_seed(args.seed + rank)
    env = create_sc2_minigame_env(args.map_name)
    game_intf = GameInterfaceHandler()

    with env:
        model = FullyConv(
            game_intf.minimap_channels,
            game_intf.screen_channels,
            game_intf.screen_resolution,
            game_intf.num_action,
            args.lstm)
        model.eval()

        state = env.reset()[0]
        reward_sum = 0
        max_score = 0

        episode_done = True
        episode_length = 0

        while True:

            if episode_done:
                model.load_state_dict(shared_model.state_dict())
                # TODO: reset lstm variables
                pass

            minimap_vb = Variable(torch.from_numpy(game_intf.get_minimap(state.observation)))
            screen_vb = Variable(torch.from_numpy(game_intf.get_screen(state.observation)))
            info_vb = Variable(torch.from_numpy(game_intf.get_info(state.observation)))
            valid_action_vb = Variable(torch.from_numpy(game_intf.get_available_actions(state.observation)), requires_grad=False)
            # TODO: if args.lstm, do model training with lstm
            value_vb, spatial_policy_vb, spatial_policy_log_vb, non_spatial_policy_vb, non_spatial_policy_log_vb, lstm_hidden_vb = model(
                minimap_vb, screen_vb, info_vb, valid_action_vb, None)
            spatial_action_ts = spatial_policy_vb.max(dim=1)[1].unsqueeze(0).data
            non_spatial_action_ts = non_spatial_policy_vb.max(dim=1)[1].unsqueeze(0).data
            sc2_action = game_intf.postprocess_action(
                non_spatial_action_ts.numpy(),
                spatial_action_ts.numpy())

            state = env.step([sc2_action])[0]  # single player
            reward = np.asscalar(state.reward)
            terminal = state.last()

            episode_done = terminal or episode_length >= args.max_episode_length

            reward_sum += reward
            episode_length += 1

            if episode_done:
                # log stats
                if summary_queue is not None:
                    summary_queue.put(
                        Summary(action='add_scalar', tag='monitor/episode_reward',
                                value1=reward_sum, global_step=global_episode_counter.value))
                    summary_queue.put(
                        Summary(action='add_scalar', tag='monitor/episode_length',
                                value1=episode_length, global_step=global_episode_counter.value))
                # save model
                if reward_sum >= max_score:
                    max_score = reward_sum
                    model_state = model.state_dict()
                    torch.save(model_state, '{0}/{1}/{2}.dat'.format(args.model_dir, args.map_name, args.job_name))
                # reset stats and env
                reward_sum = 0
                episode_length = 0
                state = env.reset()[0]  # single player
                time.sleep(5)
