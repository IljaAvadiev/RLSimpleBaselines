from algorithms.deeprl.ddpg.model import Pi, Q, DEVICE
import torch
import numpy as np
from algorithms.deeprl.td3.model import Pi, Q
from algorithms.deeprl.common.memory import ReplayBuffer
from copy import deepcopy


class TD3():
    def __init__(self, env, state_dims, action_dims, action_limit, min_action, max_action, hidden_dims, activation, optimizer,
                 pi_alpha, q_alpha, gamma, tau, max_memory_size, batch_size, dir, name):

        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.action_dims = action_dims
        self.min = min_action
        self.max = max_action
        self.memory = ReplayBuffer(state_dims, action_dims,
                                   max_memory_size=max_memory_size, batch_size=batch_size)
        self.online_policy = Pi(
            state_dims, action_dims, action_limit, hidden_dims, activation, dir, name)
        self.online_value_1 = Q(state_dims, action_dims,
                                hidden_dims, activation, dir, name)
        self.online_value_2 = Q(state_dims, action_dims,
                                hidden_dims, activation, dir, name)

        self.target_policy = deepcopy(self.online_policy)
        self.target_value_1 = deepcopy(self.online_value_1)
        self.target_value_2 = deepcopy(self.online_value_2)

        self.device = DEVICE
        self.policy_optimizer = optimizer(
            self.online_policy.parameters(), pi_alpha)
        self.value_optimizer_1 = optimizer(
            self.online_value_1.parameters(), q_alpha, weight_decay=1e-2)
        self.value_optimizer_2 = optimizer(
            self.online_value_2.parameters(), q_alpha, weight_decay=1e-2)

        # no grad calculations are required for target networks due to polyak averaging
        for params in self.target_policy.parameters():
            params.requires_grad = False
        for params in self.target_value_1.parameters():
            params.requires_grad = False
        for params in self.target_value_2.parameters():
            params.requires_grad = False

    @torch.no_grad()
    def act(self, state):
        mu = 0
        sigma = 0.1
        noise = torch.tensor(np.random.normal(
            mu, sigma, size=self.action_dims), dtype=torch.float32).to(self.device)
        state = torch.tensor(
            state, dtype=torch.float32).to(self.device)
        action = self.online_policy(state)
        action = action + noise

        return torch.clamp(action, self.min, self.max).cpu().detach().numpy()

    @torch.no_grad()
    def act_greedy(self, state):
        state = torch.tensor(
            state, dtype=torch.float32).to(self.device)
        action = self.online_policy(state)
        return action.cpu().detach().numpy()

    def sample_batch(self):
        states, actions, next_states, rewards, terminals = self.memory.sample_batch()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        terminals = torch.tensor(
            terminals, dtype=torch.float32).to(self.device)

        return states, actions, next_states, rewards, terminals

    def optimize(self, delay):
        states, actions, next_states, rewards, terminals = self.sample_batch()

        with torch.no_grad():
            next_actions = self.target_policy(next_states).detach()
            noise = torch.clamp(torch.normal(
                0, 0.2, size=next_actions.shape), -0.5, 0.5)
            next_actions += noise

            next_actions = torch.clamp(next_actions, self.min, self.max)

            target_1 = self.target_value_1(next_states, next_actions)
            target_2 = self.target_value_2(next_states, next_actions)

            target = rewards + \
                self.gamma * torch.min(target_1, target_2) * \
                torch.logical_not(terminals)

        online_1 = self.online_value_1(states, actions)
        online_2 = self.online_value_2(states, actions)

        self.value_optimizer_1.zero_grad()
        self.value_optimizer_2.zero_grad()
        loss_1 = (online_1 - target.detach()).pow(2).mul(0.5).mean()
        loss_2 = (online_2 - target.detach()).pow(2).mul(0.5).mean()
        critic_loss = loss_1 + loss_2

        critic_loss.backward()
        self.value_optimizer_1.step()
        self.value_optimizer_2.step()

        if not delay:
            self.policy_optimizer.zero_grad()
            actor_loss = -(self.online_value_1(states,
                                               self.online_policy(states)).mean())
            actor_loss.backward()
            self.policy_optimizer.step()
            self.update_target_weights()

    def update_target_weights(self):
        # polyak averaging
        with torch.no_grad():
            # update q-value target network
            for online, target in zip(self.online_value_1.parameters(), self.target_value_1.parameters()):
                target.data.mul_(1-self.tau)
                target.data.add_(self.tau * online.data)
            for online, target in zip(self.online_value_2.parameters(), self.target_value_2.parameters()):
                target.data.mul_(1-self.tau)
                target.data.add_(self.tau * online.data)
            # update policy target network
            for online, target in zip(self.online_policy.parameters(), self.target_policy.parameters()):
                target.data.mul_(1-self.tau)
                target.data.add_(self.tau * online.data)

    def save(self):
        self.actor.save()
        self.critic_1.save()
        self.critic_2.save()

    def load(self):
        self.actor.load()
        self.critic_1.save()
        self.critic_2.save()

    def learn(self, max_episodes, warmup, target_reward, log=False):
        step = 0
        rewards = []
        rewards_mean = []

        eval_rewards = []
        eval_rewards_mean = []

        for episode in range(max_episodes):
            state, done = self.env.reset(), False
            reward_sum = 0
            while not done:
                step += 1
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.add_memory(state, action, next_state, reward, done)
                state = next_state
                reward_sum += reward
                if step > warmup:
                    if step % 2 == 0:
                        delay = False
                    else:
                        delay = True
                    self.optimize(delay)

            rewards.append(reward_sum)
            eval_rewards.append(self.evaluate())
            if episode >= 100:
                mean = np.mean(rewards[-100:])
                eval_mean = np.mean(eval_rewards[-100:])
                rewards_mean.append(mean)
                eval_rewards_mean.append(eval_mean)
                if log:
                    print('--------------------------------------------------------')
                    print(f'Episode: {episode}')
                    print(f'Step: {step}')
                    print(f'Score: {reward_sum}')
                    print(f'Train Mean: {mean}')
                    print(f'Eval Score: {eval_rewards[-1]}')
                    print(f'Eval Mean 100: {eval_mean}')
                    print('--------------------------------------------------------')
                    if eval_mean >= target_reward:
                        print('GOAL ACHIEVED!')
                        return eval_rewards_mean
            else:
                print('--------------------------------------------------------')
                print(f'Episode: {episode}')

        return eval_rewards_mean

    def evaluate(self):
        state, done = self.env.reset(), False
        reward_sum = 0
        while not done:
            action = self.act_greedy(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            reward_sum += reward
        return reward_sum
