import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, state_size, action_size))

        self.alpha = 0.1
        self.gamma = 0.9

        # 🔥 epsilon decay
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def choose_action(self, state):
        x, y = state

        if random.uniform(0,1) < self.epsilon:
            return random.randint(0,3)
        else:
            return np.argmax(self.q_table[x,y])

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state

        best_next = np.max(self.q_table[nx, ny])

        self.q_table[x, y, action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[x, y, action]
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay