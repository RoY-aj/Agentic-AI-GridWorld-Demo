import numpy as np
import random

class GridWorld:
    def __init__(self, size=5, dynamic=False):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.state = self.start

        self.dynamic = dynamic

        # obstacle default (statis)
        self.obstacles = [(1,1), (2,2), (3,1)]

    def reset(self):
        self.state = self.start

        if self.dynamic:
            self.randomize_obstacles()

        return self.state

    def randomize_obstacles(self):
        self.obstacles = []

        while len(self.obstacles) < 3:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            pos = (x, y)

            # ❌ jangan nabrak start & goal
            if pos != self.start and pos != self.goal:
                # ❌ jangan duplikat & jangan ngeblok awal
                if pos not in self.obstacles and pos not in [(0,1), (1,0)]:
                    self.obstacles.append(pos)

    def step(self, action):
        x, y = self.state

        # 0=up,1=down,2=left,3=right
        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1

        # batas grid
        x = max(0, min(self.size - 1, x))
        y = max(0, min(self.size - 1, y))

        next_state = (x, y)

        # obstacle check
        if next_state in self.obstacles:
            next_state = self.state

        self.state = next_state

        # reward
        if self.state == self.goal:
            return self.state, 10, True
        else:
            return self.state, -1, False

    def get_grid(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        # goal
        gx, gy = self.goal
        grid[gx][gy] = "G"

        # obstacles
        for ox, oy in self.obstacles:
            grid[ox][oy] = "X"

        # agent
        x, y = self.state
        grid[x][y] = "A"

        return grid

    def render(self):
        grid = self.get_grid()

        for row in grid:
            print(" ".join(row))
        print("\n")