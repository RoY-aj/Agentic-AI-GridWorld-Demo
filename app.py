import streamlit as st
import matplotlib.pyplot as plt

from environment import GridWorld
from agent import QLearningAgent


# =========================
# 🔥 TRAIN FUNCTION (ADAPTIF)
# =========================
def train(env, agent, episodes):
    rewards = []
    success_count = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        # 🔥 biar tetap adaptif di dinamis
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
    avg_reward = sum(rewards) / episodes

    return rewards, success_rate, avg_reward


# =========================
# 🎨 UI
# =========================
st.title("🤖 Agentic AI GridWorld (Adaptive Version)")

st.write("Simulasi Q-Learning Agent dalam lingkungan statis dan dinamis")

# 🔥 pilih mode
mode = st.selectbox("Pilih Environment", ["Statis", "Dinamis"])

# 🔥 episode fleksibel
episodes = st.slider("Jumlah Episode", 500, 5000, 3000)


# =========================
# 🚀 RUN
# =========================
if st.button("🚀 Run Training"):

    dynamic = True if mode == "Dinamis" else False

    env = GridWorld(dynamic=dynamic)
    agent = QLearningAgent(5, 4)

    rewards, success_rate, avg_reward = train(env, agent, episodes)

    # 📈 grafik
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title(f"Learning Curve ({mode})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    st.pyplot(fig)

    # 📊 hasil
    st.subheader("📊 Hasil Pengujian")

    st.write(f"✅ Success Rate: {success_rate:.2f}%")
    st.write(f"📈 Average Reward: {avg_reward:.2f}")

    # 🧭 grid terakhir
    st.subheader("🧭 Grid Terakhir")

    grid = env.get_grid()

    for row in grid:
        st.write(" ".join(row))