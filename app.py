import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt

# =========================
# 🌍 ENVIRONMENT
# =========================
class GridWorld:
    def __init__(self, size=5, dynamic=False):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.state = self.start
        self.dynamic = dynamic
        self.obstacles = [(1,1), (2,2), (3,1)]

    def reset(self):
        self.state = self.start
        if self.dynamic:
            self.randomize_obstacles()
        return self.state

    def randomize_obstacles(self):
        self.obstacles = []
        while len(self.obstacles) < 3:
            pos = (random.randint(0,4), random.randint(0,4))
            if pos != self.start and pos != self.goal:
                if pos not in self.obstacles:
                    self.obstacles.append(pos)

    def step(self, action):
        x, y = self.state

        if action == 0: x -= 1
        elif action == 1: x += 1
        elif action == 2: y -= 1
        elif action == 3: y += 1

        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))

        next_state = (x, y)

        if next_state in self.obstacles:
            next_state = self.state

        self.state = next_state

        if self.state == self.goal:
            return self.state, 10, True
        else:
            return self.state, -1, False

    def get_grid(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        gx, gy = self.goal
        grid[gx][gy] = "G"

        for ox, oy in self.obstacles:
            grid[ox][oy] = "X"

        x, y = self.state
        grid[x][y] = "A"

        return grid


# =========================
# 🤖 AGENT
# =========================
class QLearningAgent:
    def __init__(self, size, action_size):
        self.q_table = np.zeros((size, size, action_size))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def choose_action(self, state):
        x, y = state
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0,3)
        return np.argmax(self.q_table[x,y])

    def update(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state

        best_next = np.max(self.q_table[nx, ny])

        self.q_table[x, y, action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[x, y, action]
        )


# =========================
# 🔥 TRAIN
# =========================
def train(env, agent, episodes):
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0

        for _ in range(200):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    return rewards


# =========================
# 🎨 STREAMLIT UI
# =========================
st.title("Agentic AI GridWorld Demo 🤖")

mode = st.selectbox("Pilih Environment", ["Statis", "Dinamis"])
episodes = st.slider("Jumlah Episode", 500, 3000, 1500)

if st.button("🚀 Run Training"):

    dynamic = True if mode == "Dinamis" else False

    env = GridWorld(dynamic=dynamic)
    agent = QLearningAgent(5, 4)

    rewards = train(env, agent, episodes)

    # 📊 Grafik
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title("Learning Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    st.pyplot(fig)

    # 🎮 Visual akhir
    st.subheader("Visualisasi Agent")

    state = env.reset()

    for _ in range(20):
        grid = env.get_grid()
        st.text("\n".join([" ".join(row) for row in grid]))

        action = agent.choose_action(state)
        state, _, done = env.step(action)

        if done:
            st.success("🎯 Goal tercapai!")
            break