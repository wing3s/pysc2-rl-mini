import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from envs import create_sc2_minigame_env
from envs import GameInterfaceHandler
from model import FullyConv
from summary import Summary


def ensure_shared_grads(model, shared_model):
    """ ensure proper initialization of global grad"""
    # NOTE: due to no backward passes has ever been ran on the global model
    # NOTE: ref: https://discuss.pytorch.org/t/problem-on-variable-grad-data/957
    for shared_param, local_param in zip(shared_model.parameters(),
                                         model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = local_param.grad  # pylint: disable=W0212


def worker_fn(rank, args, shared_model, global_counter, summary_queue, optimizer):
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
        model.train()

        state = env.reset()[0]  # state as TimeStep object from single player

        episode_done = True
        episode_length = 0

        # TODO: verify stop condition
        while True:
            # Sync from the global shared model
            model.load_state_dict(shared_model.state_dict())

            # start a new episode
            if episode_done:
                # TODO: reset lstm variables
                pass

            # reset observed variables
            entropies = []
            value_vbs = []
            spatial_policy_log_for_action_vbs = []
            non_spatial_policy_log_for_action_vbs = []
            rewards = []

            # rollout, step forward n steps
            for step in range(args.num_forward_steps):
                minimap_vb = Variable(torch.from_numpy(game_intf.get_minimap(state.observation)))
                screen_vb = Variable(torch.from_numpy(game_intf.get_screen(state.observation)))
                info_vb = Variable(torch.from_numpy(game_intf.get_info(state.observation)))
                valid_action_vb = Variable(torch.from_numpy(game_intf.get_available_actions(state.observation)), requires_grad=False)
                # TODO: if args.lstm, do model training with lstm
                value_vb, spatial_policy_vb, spatial_policy_log_vb, non_spatial_policy_vb, non_spatial_policy_log_vb, lstm_hidden_vb = model(
                    minimap_vb, screen_vb, info_vb, valid_action_vb, None)
                # Entropy of a probability distribution is the expected value of - log P(X),
                # computed as sum(policy * -log(policy)) which is positive.
                # Entropy is smaller when the probability distribution is more centered on one action
                # so larger entropy implies more exploration.
                # Thus we penalise small entropy which is adding -entropy to our loss.
                spatial_entropy = -(spatial_policy_log_vb * spatial_policy_vb).sum(1)
                non_spatial_entropy = -(non_spatial_policy_log_vb * non_spatial_policy_vb).sum(1)
                entropy = spatial_entropy + non_spatial_entropy
                entropies.append(entropy)

                spatial_action_ts = spatial_policy_vb.multinomial().data
                non_spatial_action_ts = non_spatial_policy_vb.multinomial().data
                sc2_action = game_intf.postprocess_action(
                    non_spatial_action_ts.numpy(),
                    spatial_action_ts.numpy())
                # For a given state and action, compute the log of the policy at
                # that action for that state.
                spatial_policy_log_for_action_vb = spatial_policy_log_vb.gather(1, Variable(spatial_action_ts))
                non_spatial_policy_log_for_action_vb = non_spatial_policy_log_vb.gather(1, Variable(non_spatial_action_ts))

                state = env.step([sc2_action])[0]  # single player
                reward = np.asscalar(state.reward)
                terminal = state.last()

                episode_done = terminal or episode_length >= args.max_episode_length

                value_vbs.append(value_vb)
                spatial_policy_log_for_action_vbs.append((spatial_policy_log_for_action_vb))
                non_spatial_policy_log_for_action_vbs.append((non_spatial_policy_log_for_action_vb))
                rewards.append(reward)

                episode_length += 1
                global_counter.value += 1

                if episode_done:
                    episode_length = 0
                    state = env.reset()[0]
                    break

            # R: estimate reward based on policy pi
            R_ts = torch.zeros(1, 1)
            if not episode_done:
                # bootstrap from last state
                # TODO: if args.lstm
                minimap_vb = Variable(
                    torch.from_numpy(game_intf.get_minimap(state.observation)))
                screen_vb = Variable(
                    torch.from_numpy(game_intf.get_screen(state.observation)))
                info_vb = Variable(
                    torch.from_numpy(game_intf.get_info(state.observation)))
                valid_action_vb = Variable(
                    torch.from_numpy(game_intf.get_available_actions(state.observation)))
                value_vb, _, _, _, _, _ = model(minimap_vb, screen_vb, info_vb, valid_action_vb, None)
                R_ts = value_vb.data

            R_vb = Variable(R_ts)
            value_vbs.append(R_vb)

            policy_loss_vb = 0.
            value_loss_vb = 0.
            gae_ts = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                R_vb = args.gamma * R_vb + rewards[i]
                advantage_vb = R_vb - value_vbs[i]
                value_loss_vb += 0.5 * advantage_vb.pow(2)

                # Generalized Advantage Estimation
                # Refer to http://www.breloff.com/DeepRL-OnlineGAE
                # equation 16, 18
                # tderr_ts: Discounted sum of TD residuals
                tderr_ts = rewards[i] + args.gamma * value_vbs[i+1].data - value_vbs[i].data
                gae_ts = gae_ts * args.gamma * args.tau + tderr_ts

                # Try to do gradient ascent on the expected discounted reward
                # The gradient of the expected discounted reward is the gradient
                # of log pi * (R - estimated V), where R is the sampled reward
                # from the given state following the policy pi.
                # Since we want to max this value, we define policy loss as negative
                # NOTE: the negative entropy term  encourages exploration
                policy_log_for_action_vb = spatial_policy_log_for_action_vbs[i] + non_spatial_policy_log_for_action_vbs[i]
                policy_loss_vb += -(policy_log_for_action_vb * Variable(gae_ts) + 0.1 * entropies[i])

            optimizer.zero_grad()

            loss_vb = policy_loss_vb + 0.5 * value_loss_vb
            loss_vb.backward()

            # prevent gradient explosion
            torch.nn.utils.clip_grad_norm(model.parameters(), 40)
            ensure_shared_grads(model, shared_model)

            optimizer.step()

            # log stats
            if summary_queue is not None:
                summary_queue.put(
                    Summary(action='add_scalar', tag='train/policy_loss',
                            value1=policy_loss_vb[0][0], global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_scalar', tag='train/value_loss',
                            value1=value_loss_vb[0][0], global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_scalar', tag='train/rewards/sum',
                            value1=np.array(rewards).sum(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_scalar', tag='train/entropies/mean',
                            value1=np.array(entropies).mean(), global_step=global_counter.value))

                summary_queue.put(
                    Summary(action='add_histogram', tag='policy/spatial_vb)',
                            value1=spatial_policy_vb.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='policy/non_spatial_vb',
                            value1=non_spatial_policy_vb.data.numpy(), global_step=global_counter.value))

                summary_queue.put(
                    Summary(action='add_histogram', tag='model/mconv1_weight',
                            value1=model.mconv1.weight.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/mconv1_bias',
                            value1=model.mconv1.bias.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/mconv2_weight',
                            value1=model.mconv2.weight.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/mconv2_bias',
                            value1=model.mconv2.bias.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sconv1_weight',
                            value1=model.sconv1.weight.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sconv1_bias',
                            value1=model.sconv1.bias.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sconv2_weight',
                            value1=model.sconv2.weight.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sconv2_bias',
                            value1=model.sconv2.bias.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sa_conv3_weight',
                            value1=model.sa_conv3.weight.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sa_conv3_bias',
                            value1=model.sa_conv3.bias.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/ns_fc3_weight',
                            value1=model.ns_fc3.weight.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/ns_fc3_bias',
                            value1=model.ns_fc3.bias.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/nsa_fc4_weight',
                            value1=model.nsa_fc4.weight.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/nsa_fc4_bias',
                            value1=model.nsa_fc4.bias.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/nsc_fc4_weight',
                            value1=model.nsc_fc4.weight.data.numpy(), global_step=global_counter.value))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/nsc_fc4_bias',
                            value1=model.nsc_fc4.bias.data.numpy(), global_step=global_counter.value))
