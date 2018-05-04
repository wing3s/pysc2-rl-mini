import numpy as np
import torch
from torch.autograd import Variable

from envs import create_sc2_minigame_env
from envs import GameInterfaceHandler
from model import FullyConv
from summary import Summary
from utils.gpu import cuda


def ensure_shared_grads(model, shared_model, gpu_id):
    """ ensure proper initialization of global grad"""
    # NOTE: due to no backward passes has ever been ran on the global model
    # NOTE: ref: https://discuss.pytorch.org/t/problem-on-variable-grad-data/957
    for shared_param, local_param in zip(shared_model.parameters(),
                                         model.parameters()):
        if gpu_id >= 0:
            # GPU
            shared_param._grad = local_param.grad.clone().cpu()  # pylint: disable=W0212
        else:
            # CPU
            if shared_param.grad is not None:
                return
            else:
                shared_param._grad = local_param.grad  # pylint: disable=W0212


def worker_fn(rank, args, shared_model, global_episode_counter, summary_queue, optimizer):
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    torch.manual_seed(args.seed + rank)
    summary_iters = args.summary_iters
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
        summary_iters *= 5  # send stats less frequent with GPU

    env = create_sc2_minigame_env(args.map_name, args.mode)
    game_intf = GameInterfaceHandler(args.mode)

    with env:
        model = FullyConv(
            game_intf.minimap_channels,
            game_intf.screen_channels,
            game_intf.screen_resolution,
            game_intf.num_action,
            args.lstm)
        cuda(model, gpu_id)
        model.train()

        state = env.reset()[0]  # state as TimeStep object from single player

        episode_done = True
        episode_length = 0
        local_update_count = 0

        # TODO: verify stop condition
        while True:
            if 0 < args.max_k_episode * 1000 <= global_episode_counter.value:
                break

            # Sync from the global shared model
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    model.load_state_dict(shared_model.state_dict())
            else:
                model.load_state_dict(shared_model.state_dict())

            # start a new episode
            if episode_done:
                # TODO: reset lstm variables
                pass

            # reset observed variables
            entropies = []
            spatial_entropies = []
            non_spatial_entropies = []
            value_vbs = []
            spatial_policy_log_for_action_vbs = []
            non_spatial_policy_log_for_action_vbs = []
            rewards = []
            select_spatial_acts = []

            # rollout, step forward n steps
            for step in range(args.num_forward_steps):
                minimap_vb = Variable(
                    cuda(torch.from_numpy(game_intf.get_minimap(state.observation)), gpu_id))
                screen_vb = Variable(
                    cuda(torch.from_numpy(game_intf.get_screen(state.observation)), gpu_id))
                info_vb = Variable(
                    cuda(torch.from_numpy(game_intf.get_info(state.observation)), gpu_id))
                valid_action_vb = Variable(
                    cuda(torch.from_numpy(game_intf.get_available_actions(state.observation)), gpu_id))
                # TODO: if args.lstm, do model training with lstm
                value_vb, spatial_policy_vb, non_spatial_policy_vb, lstm_hidden_vb = model(
                    minimap_vb, screen_vb, info_vb, valid_action_vb, None)

                # sample and select action
                spatial_action_ts = spatial_policy_vb.multinomial(1).data
                non_spatial_action_ts = non_spatial_policy_vb.multinomial(1).data
                sc2_action = game_intf.postprocess_action(
                    non_spatial_action_ts.cpu().numpy(),
                    spatial_action_ts.cpu().numpy())

                select_spatial_act = float(game_intf.is_non_spatial_action(sc2_action.function))
                select_spatial_acts.append(select_spatial_act)

                # Entropy of a probability distribution is the expected value of - log P(X),
                # computed as sum(policy * -log(policy)) which is positive.
                # Entropy is smaller when the probability distribution is more centered on one action
                # so larger entropy implies more exploration.
                # Thus we penalise small entropy which is adding -entropy to our loss.
                spatial_entropy = -(
                    torch.log(torch.clamp(spatial_policy_vb, min=1e-12)) *
                    spatial_policy_vb).sum(1)  # avoid log(0)
                spatial_entropy *= select_spatial_act
                non_spatial_entropy = -(
                    torch.log(torch.clamp(non_spatial_policy_vb, min=1e-12)) *
                    non_spatial_policy_vb).sum(1)  # avoid log(0)
                entropy = spatial_entropy + non_spatial_entropy
                entropies.append(entropy)
                spatial_entropies.append(spatial_entropy)
                non_spatial_entropies.append(non_spatial_entropy)

                # For a given state and action, compute the log of the policy at
                # that action for that state.
                spatial_policy_log_for_action_vb = torch.log(spatial_policy_vb.gather(1, Variable(spatial_action_ts)))
                spatial_policy_log_for_action_vb *= select_spatial_act  # set to 0 if non-spatial action is chosen
                non_spatial_policy_log_for_action_vb = torch.log(non_spatial_policy_vb.gather(1, Variable(non_spatial_action_ts)))

                state = env.step([sc2_action])[0]  # single player
                reward = np.asscalar(state.reward)
                terminal = state.last()

                episode_done = terminal or episode_length >= args.max_episode_length

                value_vbs.append(value_vb)
                spatial_policy_log_for_action_vbs.append((spatial_policy_log_for_action_vb))
                non_spatial_policy_log_for_action_vbs.append((non_spatial_policy_log_for_action_vb))
                rewards.append(reward)

                episode_length += 1

                if episode_done:
                    global_episode_counter.value += 1
                    episode_length = 0
                    state = env.reset()[0]
                    break

            # R: estimate reward based on policy pi
            R_ts = torch.zeros(1, 1)
            if not episode_done:
                # bootstrap from last state
                # TODO: if args.lstm
                minimap_vb = Variable(
                    cuda(torch.from_numpy(game_intf.get_minimap(state.observation)), gpu_id))
                screen_vb = Variable(
                    cuda(torch.from_numpy(game_intf.get_screen(state.observation)), gpu_id))
                info_vb = Variable(
                    cuda(torch.from_numpy(game_intf.get_info(state.observation)), gpu_id))
                valid_action_vb = Variable(
                    cuda(torch.from_numpy(game_intf.get_available_actions(state.observation)), gpu_id))
                value_vb, _, _, _ = model(minimap_vb, screen_vb, info_vb, valid_action_vb, None)
                R_ts = value_vb.data

            R_vb = Variable(cuda(R_ts, gpu_id))
            value_vbs.append(R_vb)

            policy_loss_vb = 0.
            value_loss_vb = 0.
            gae_ts = cuda(torch.zeros(1, 1), gpu_id)
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
                policy_loss_vb += -(policy_log_for_action_vb * Variable(gae_ts) + args.entropy_weight * entropies[i])

            optimizer.zero_grad()

            loss_vb = policy_loss_vb + 0.5 * value_loss_vb
            loss_vb.backward()

            # prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            ensure_shared_grads(model, shared_model, gpu_id)

            optimizer.step()
            local_update_count += 1

            # log scalar stats
            if summary_queue is not None and local_update_count % summary_iters == 0:
                global_episode_counter_val = global_episode_counter.value
                counter_f_path = '{0}/{1}/{2}/{3}/counter.log'.format(args.log_dir, args.mode, args.map_name, args.job_name)
                with open(counter_f_path, 'w') as counter_f:
                    counter_f.write(str(global_episode_counter_val))
                # loss
                summary_queue.put(
                    Summary(action='add_scalar', tag='loss/policy',
                            value1=policy_loss_vb.cpu()[0][0], global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_scalar', tag='loss/value',
                            value1=value_loss_vb.cpu()[0][0], global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_scalar', tag='loss/total',
                            value1=loss_vb.cpu()[0][0], global_step=global_episode_counter_val))
                # reward
                summary_queue.put(
                    Summary(action='add_scalar', tag='train/rewards/sum',
                            value1=np.array(rewards).sum(), global_step=global_episode_counter_val))
                # entropy
                summary_queue.put(
                    Summary(action='add_scalar', tag='entropy/total/mean',
                            value1=torch.cat(entropies, 0).cpu().mean(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(
                        action='add_scalar',
                        tag='entropy/spatial/mean',
                        value1=(
                            torch.cat(spatial_entropies, 0).cpu().sum() /
                            max(np.array(select_spatial_acts).sum(), 1e-12)
                        ),
                        global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_scalar', tag='entropy/non_spatial/mean',
                            value1=torch.cat(non_spatial_entropies, 0).cpu().mean(), global_step=global_episode_counter_val))

                # value
                summary_queue.put(
                    Summary(action='add_scalar', tag='value/estimate/mean',
                            value1=torch.cat(value_vbs, 0).cpu().mean(), global_step=global_episode_counter_val))
                # action log probability
                summary_queue.put(
                    Summary(
                        action='add_scalar',
                        tag='action/selected_log_prob/total/mean',
                        value1=torch.cat(spatial_policy_log_for_action_vbs +
                                         non_spatial_policy_log_for_action_vbs,
                                         0).cpu().mean(),
                        global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(
                        action='add_scalar',
                        tag='action/selected_log_prob/spatial/mean',
                        value1=(
                            torch.cat(spatial_policy_log_for_action_vbs, 0).cpu().sum() /
                            max(np.array(select_spatial_acts).sum(), 1e-12)
                        ),
                        global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(
                        action='add_scalar',
                        tag='action/selected_log_prob/non_spatial/mean',
                        value1=torch.cat(non_spatial_policy_log_for_action_vbs,
                                         0).cpu().mean(),
                        global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(
                        action='add_scalar',
                        tag='action/selected_spatial_action/mean',
                        value1=np.array(select_spatial_acts).mean(),
                        global_step=global_episode_counter_val))

            # log distribution stats
            if summary_queue is not None and local_update_count % (summary_iters * 10) == 0:
                global_episode_counter_val = global_episode_counter.value
                summary_queue.put(
                    Summary(action='add_histogram', tag='policy/spatial_vb',
                            value1=spatial_policy_vb.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='policy/non_spatial_vb',
                            value1=non_spatial_policy_vb.data.cpu().numpy(), global_step=global_episode_counter_val))

                summary_queue.put(
                    Summary(action='add_histogram', tag='model/mconv1_weight',
                            value1=model.mconv1.weight.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/mconv1_bias',
                            value1=model.mconv1.bias.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/mconv2_weight',
                            value1=model.mconv2.weight.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/mconv2_bias',
                            value1=model.mconv2.bias.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sconv1_weight',
                            value1=model.sconv1.weight.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sconv1_bias',
                            value1=model.sconv1.bias.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sconv2_weight',
                            value1=model.sconv2.weight.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sconv2_bias',
                            value1=model.sconv2.bias.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/sa_conv3_weight',
                            value1=model.sa_conv3.weight.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/ns_fc3_weight',
                            value1=model.ns_fc3.weight.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/ns_fc3_bias',
                            value1=model.ns_fc3.bias.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/nsa_fc4_weight',
                            value1=model.nsa_fc4.weight.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/nsa_fc4_bias',
                            value1=model.nsa_fc4.bias.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_histogram', tag='model/nsc_fc4_weight',
                            value1=model.nsc_fc4.weight.data.cpu().numpy(), global_step=global_episode_counter_val))

                summary_queue.put(
                    Summary(action='add_scalar', tag='model/sa_conv3_bias',
                            value1=model.sa_conv3.bias.data.cpu().numpy(), global_step=global_episode_counter_val))
                summary_queue.put(
                    Summary(action='add_scalar', tag='model/nsc_fc4_bias',
                            value1=model.nsc_fc4.bias.data.cpu().numpy(), global_step=global_episode_counter_val))
