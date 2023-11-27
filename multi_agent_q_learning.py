from pettingzoo.mpe import simple_spread_v2
import numpy as np
import random
from collections import deque

# Create the multi-agent environment
env = simple_spread_v2.env(N=3, local_ratio=0.5)
env.reset()
num_agents = env.num_agents

# Hyperparameters
total_episodes = 1000
learning_rate = 0.1
max_steps = 100
gamma = 0.95  # Discount rate

# Exploration parameters
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

# Initialize Q-tables for each agent
# Note: The state and action spaces are determined by the specific environment
state_size = env.observation_space(env.agents[0]).shape[0]
action_size = env.action_space(env.agents[0]).n
Q = [np.zeros((state_size, action_size)) for _ in range(num_agents)]

# Replay buffer for each agent
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def store(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size=64):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

replay_buffers = [ReplayBuffer() for _ in range(num_agents)]

# Function to choose the action for each agent
def choose_action(state, agent_id):
    if random.uniform(0, 1) < epsilon:
        return np.random.randint(action_size)  # Explore
    else:
        return np.argmax(Q[agent_id][state])  # Exploit

# Function to learn from experience for each agent
def learn_from_experience(agent_id, batch_size):
    mini_batch = replay_buffers[agent_id].sample(batch_size)
    for experience in mini_batch:
        state, action, reward, next_state, done = experience
        predict = Q[agent_id][state][action]
        target = reward if done else reward + gamma * np.max(Q[agent_id][next_state])
        Q[agent_id][state][action] = predict + learning_rate * (target - predict)

# Main loop
for episode in range(total_episodes):
    env.reset()
    total_rewards = [0 for _ in range(num_agents)]
    states = {agent: env.observe(agent) for agent in env.agents}

    for step in range(max_steps):
        actions = {agent: choose_action(states[agent], env.agents.index(agent)) for agent in env.agents}
        env.step(actions)
        next_states = {agent: env.observe(agent) for agent in env.agents}
        rewards = env.rewards
        dones = env.dones

        # Store experiences in the replay buffer and learn
        for i, agent in enumerate(env.agents):
            replay_buffers[i].store((states[agent], actions[agent], rewards[i], next_states[agent], dones[i]))
            if len(replay_buffers[i].buffer) > 64:
                learn_from_experience(i, 64)

        states = next_states
        if all(dones.values()):
            break

    # Decay epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("Training finished.\n")

# Test the agents
for episode in range(5):
    env.reset()
    states = {agent: env.observe(agent) for agent in env.agents}
    for step in range(max_steps):
        actions = {agent: np.argmax(Q[env.agents.index(agent)][states[agent]]) for agent in env.agents}
        env.step(actions)
        next_states = {agent: env.observe(agent) for agent in env.agents}
        states = next_states
        env.render()
        if all(env.dones.values()):
            break
    print(f"Episode {episode + 1} complete")

env.close()
