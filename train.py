import torch
from torch.autograd import Variable

from envs import create_sc2_minigame_env
from envs import GameInterfaceHandler
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    """ ensure proper initialization of global grad"""
    # NOTE: due to no backward passes has ever been ran on the global model
    # NOTE: ref: https://discuss.pytorch.org/t/problem-on-variable-grad-data/957
    for shared_param, local_param in zip(shared_model.parameters(),
                                         model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = local_param.grad  # pylint: disable=W0212


def train_fn(idx, args, shared_model, global_counter, optimizer):
    torch.manual_seed(args.seed + idx)
    env = create_sc2_minigame_env(args.map_name)

    game_intf = GameInterfaceHandler()
    model = ActorCritic(
        game_intf.minimap_channels,
        game_intf.screen_channels,
        game_intf.screen_resolution,
        game_intf.num_action,
        args.lstm)
    model.train()

    state = env.reset()  # numpy array

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
        policy_log_for_action_vbs = []
        rewards = []

        # rollout, step forward n steps
        for step in range(args.num_forward_steps):
            if args.lstm:
                value_vb, policy_vb, lstm_hidden_vb = model(
                    get_state_vb(state), lstm_hidden_vb)
            else:
                value_vb, policy_vb, _ = model(get_state_vb(state))

            # Entropy of a probability distribution is the expected value of - log P(X),
            # computed as sum(policy * -log(policy)) which is positive.
            # Entropy is smaller when the probability distribution is more centered on one action
            # so larger entropy implies more exploration.
            # Thus we penalise small entropy which is adding -entropy to our loss.
            policy_log_vb = torch.log(policy_vb)
            entropy = -(policy_log_vb * policy_vb).sum(1)
            entropies.append(entropy)

            action_ts = policy_vb.multinomial().data
            # For a given state and action, compute the log of the policy at
            # that action for that state.
            policy_log_for_action_vb = policy_log_vb.gather(1, Variable(action_ts))

            state, reward, terminal, _ = env.step(action_ts.numpy())

            episode_done = terminal or episode_length >= args.max_episode_length

            value_vbs.append(value_vb)
            policy_log_for_action_vbs.append(policy_log_for_action_vb)
            rewards.append(reward)

            episode_length += 1
            global_counter.value += 1

            if episode_done:
                episode_length = 0
                state = env.reset()
                break

        # R: estimate reward based on policy pi
        R_ts = torch.zeros(1, 1)
        if not episode_done:
            # bootstrap from last state
            if args.lstm:
                value_vb, _, _ = model(get_state_vb(state), lstm_hidden_vb)
            else:
                value_vb, _, _ = model(get_state_vb(state))
            R_ts = value_vb.data

        R_vb = Variable(R_ts)
        value_vbs.append(R_vb)

        policy_loss_vb = 0.
        value_loss_vb = 0.
        gae_ts = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R_vb = args.gamma * R_vb + reward[i]
            advantage_vb = R_vb - value_vbs[i]
            value_loss_vb += 0.5 * advantage_vb.pow(2)

            # Generalized Advantage Estimation
            # Refer to http://www.breloff.com/DeepRL-OnlineGAE
            # equation 16, 18
            # tderr_ts: Discounted sum of TD residuals
            tderr_ts = reward[i] + args.gamma * value_vbs[i+1].data - value_vbs[i].data
            gae_ts = gae_ts * args.gamma * args.tau + tderr_ts

            # Try to do gradient ascent on the expected discounted reward
            # The gradient of the expected discounted reward is the gradient
            # of log pi * (R - estimated V), where R is the sampled reward
            # from the given state following the policy pi.
            # Since we want to max this value, we define policy loss as negative
            # NOTE: the negative entropy term  encourages exploration
            policy_loss_vb += -(policy_log_for_action_vbs[i] * Variable(gae_ts) + 0.01 * entropies[i])

        optimizer.zero_grad()

        loss_vb = policy_loss_vb + 0.5 * value_loss_vb
        loss_vb.backward()

        # prevent gradient explosion
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)

        optimizer.step()


def get_state_vb(state):
    """Convert state from numpy array to variable"""
    return Variable(torch.from_numpy(state).unsqueeze(0))
