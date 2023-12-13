import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from collections import deque
import random

def save_optimal_policy(optimal_policy, filename='optimal_policy.csv'):
    print(optimal_policy.shape)
    with open(filename, 'w') as file:
        for i in range(optimal_policy.shape[2]):
            for j in range(optimal_policy.shape[0]):
                for k in range(optimal_policy.shape[1]):
                    file.write(f"{j},{k},{i + 1},{optimal_policy[j][k][i] + 1}\n")
    print(f"Optimal policy with queue saved to {filename}")

def calculate_optimal_policy(agent):
    optimal_policy = np.zeros((31, 31, 2), dtype=int)
    for i in range(31):
        for j in range(31):
            for k in range(2):
                state = np.array([[i, j, k]])
                action = agent.act(state)
                optimal_policy[i][j][k] = action
    return optimal_policy

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

EPISODES = 2
queue_1_size = 0
queue_2_size = 0
batch_size = 32

state_size = 3
action_size = 2

agent = DQNAgent(state_size, action_size)

for e in range(EPISODES):
    state = np.array([[queue_1_size, queue_2_size, 0]])

    for time in range(100):
        action = agent.act(state)
        
        if action == 0:
            state[0][2] = 1 - state[0][2]
            reward = 0
        else:
            if state[0][2] == 0:
                if queue_1_size > 0:
                    queue_1_size -= 1
                    reward = 1
                else:
                    reward = 0
            else:
                if queue_2_size > 0:
                    queue_2_size -= 1
                    reward = 1
                else:
                    reward = 0
        
        if np.random.rand() <= 0.1 and queue_1_size < 30:
            queue_1_size += 1
        if np.random.rand() <= 0.7 and queue_2_size < 30:
            queue_2_size += 1
        
        next_state = np.array([[queue_1_size, queue_2_size, state[0][2]]])
        
        done = False
        
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        
        if done:
            break
        
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    
    print(f"Episode: {e + 1}/{EPISODES}, Exploration Rate: {agent.epsilon}")

agent.model.save("trained_model.keras")
print("Model saved successfully")

optimal_policy = calculate_optimal_policy(agent)
print("Optimal Policy with Queue:")
print(optimal_policy)


save_optimal_policy(optimal_policy, 'optimal_policy.csv')
