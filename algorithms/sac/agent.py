import torch
from model import Actor, Critic
from replay_buffer import ReplayBuffer
from copy import deepcopy


# highly inspired by the openai spinningup implementation
class Agent():
    def __init__(self, state_dims, action_dims, hidden_dims, lr_actor, lr_critic, alpha, gamma, tau, mem_size, batch_size, log_sigma_min, log_sigma_max, warmup):
        self.memory = ReplayBuffer(
            state_dims, action_dims, max_memsize=mem_size, batch_size=batch_size)

        self.actor = Actor(state_dims, action_dims, hidden_dims,
                           lr_actor, log_sigma_min, log_sigma_max)

        self.critic_1 = Critic(state_dims, action_dims, hidden_dims, lr_critic)
        self.critic_2 = Critic(state_dims, action_dims, hidden_dims, lr_critic)

        self.actor_target = deepcopy(self.actor)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)

        self.counter = 0
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.warmup = warmup

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add_memory(state, action, reward, next_state, done)

    def choose_action(self, state):
        self.counter += 1
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(
                self.actor.device).unsqueeze(0)
            action, _ = self.actor.forward(state)
            return action.cpu().detach().numpy().reshape(-1)

    def greedy_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(
                self.actor.device).unsqueeze(0)
            action, _ = self.actor.forward(state, greedy=True)
            return action.cpu().detach().numpy().reshape(-1)

    def calc_loss_q(self, states, actions, rewards, next_states, terminals):
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        with torch.no_grad():
            next_actions, log_probs = self.actor(next_states)
            q1_target = self.critic_1_target(next_states, next_actions)
            q2_target = self.critic_2_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            target = rewards + self.gamma * \
                torch.logical_not(terminals) * \
                (q_target - self.alpha * log_probs)

        loss_1 = ((q1 - target)**2).mean()
        loss_2 = ((q2 - target)**2).mean()

        loss = loss_1 + loss_2
        return loss

    def calc_loss_pi(self, states):
        actions, log_probs = self.actor(states)
        q1 = self.critic_1_target(states, actions)
        q2 = self.critic_2_target(states, actions)

        q = torch.min(q1, q2)
        loss = -(q - self.alpha * log_probs).mean()

        return loss

    def learn(self):

        if self.counter < self.warmup:
            return

        device = self.actor.device
        states, actions, rewards, next_states, terminals = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        terminals = torch.tensor(next_states, dtype=torch.float32).to(device)

        loss_q = self.calc_loss_q(
            states, actions, rewards, next_states, terminals)

        loss_pi = self.calc_loss_pi(states)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        loss_q.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.actor.optimizer.zero_grad()
        loss_pi.backward()
        self.actor.optimizer.step()
        self.update_target_weights()

    def update_target_weights(self):
        actor_params = self.actor.named_parameters()
        actor_target_params = self.actor_target.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        critic_1_target_params = self.critic_1_target.named_parameters()
        critic_2_target_params = self.critic_2_target.named_parameters()

        actor_state_dict = dict(actor_params)
        actor_target_state_dict = dict(actor_target_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        critic_1_target_state_dict = dict(critic_1_target_params)
        critic_2_target_state_dict = dict(critic_2_target_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = self.tau*critic_1_state_dict[name].clone() + \
                (1-self.tau)*critic_1_target_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = self.tau*critic_2_state_dict[name].clone() + \
                (1-self.tau)*critic_2_target_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = self.tau*actor_state_dict[name].clone() + \
                (1-self.tau)*actor_target_state_dict[name].clone()

        self.critic_1_target.load_state_dict(critic_1_state_dict)
        self.critic_2_target.load_state_dict(critic_2_state_dict)
        self.actor_target.load_state_dict(actor_state_dict)
