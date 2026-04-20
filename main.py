from environment import GridWorld
from agent import QLearningAgent
import matplotlib.pyplot as plt
import time
import os


def train(env, agent, episodes):
    rewards = []
    success_count = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        # 🔥 reset epsilon biar tetap adaptif
        agent.epsilon = max(agent.epsilon, 0.3)

        for step in range(100):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                success_count += 1
                break

        rewards.append(total_reward)

    success_rate = (success_count / episodes) * 100
    return rewards, success_rate


# 🔥 visual greedy (hasil belajar)
def visualize(env, agent):
    state = env.reset()

    for step in range(30):
        os.system('cls' if os.name == 'nt' else 'clear')

        print(f"Step {step+1}")
        env.render()

        time.sleep(0.3)

        x, y = state
        action = agent.q_table[x, y].argmax()

        next_state, _, done = env.step(action)

        state = next_state

        if done:
            print("\n🎯 GOAL TERCAPAI!")
            break


if __name__ == "__main__":

    episodes = 100

    # 🔥 DINAMIS TEST
    env = GridWorld(dynamic=False)
    agent = QLearningAgent(5, 4)

    rewards, success_rate = train(env, agent, episodes)

    print("\n=== HASIL DINAMIS ===")
    print(f"Success Rate: {success_rate:.2f}%")

    # 📈 grafik harus lebih “terkendali”
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve (Dinamis - Adaptif)")
    plt.show()

    print("\n=== VISUALISASI ===")
    visualize(env, agent)