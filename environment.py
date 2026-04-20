import random

class GridWorld:
    def __init__(self, size=5, dynamic=False):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = self.start
        self.dynamic = dynamic

        # obstacle awal
        self.obstacles = [(1,1), (2,2), (3,1)]

    def reset(self):
        self.state = self.start

        # 🔥 obstacle tidak selalu berubah
        if self.dynamic and random.random() < 0.3:
            self.randomize_obstacles()

        return self.state

    def randomize_obstacles(self):
        self.obstacles = []

        while len(self.obstacles) < 3:
            pos = (random.randint(0,4), random.randint(0,4))

            if pos != self.start and pos != self.goal:
                if pos not in self.obstacles and pos not in [(0,1),(1,0)]:
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

        # 🔥 obstacle penalty
        if next_state in self.obstacles:
            next_state = self.state
            reward = -1
        else:
            reward = -1

        # 🔥 reward shaping (biar ada arah ke goal)
        old_distance = abs(self.state[0] - self.goal[0]) + abs(self.state[1] - self.goal[1])
        new_distance = abs(next_state[0] - self.goal[0]) + abs(next_state[1] - self.goal[1])

        if new_distance < old_distance:
            reward += 0.5

        self.state = next_state

        if self.state == self.goal:
            return self.state, 10, True
        else:
            return self.state, reward, False

    # 🎮 buat terminal
    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        gx, gy = self.goal
        grid[gx][gy] = "G"

        for ox, oy in self.obstacles:
            grid[ox][oy] = "X"

        x, y = self.state
        grid[x][y] = "A"

        print("\n")
        for row in grid:
            row_str = ""
            for cell in row:
                if cell == "A":
                    row_str += "🟦 "
                elif cell == "G":
                    row_str += "🟨 "
                elif cell == "X":
                    row_str += "🟥 "
                else:
                    row_str += "⬜ "
            print(row_str)

    # 🌐 buat web (Streamlit)
    def get_grid(self):
        grid = [["⬜" for _ in range(self.size)] for _ in range(self.size)]

        gx, gy = self.goal
        grid[gx][gy] = "🟨"

        for ox, oy in self.obstacles:
            grid[ox][oy] = "🟥"

        x, y = self.state
        grid[x][y] = "🟦"

        return grid