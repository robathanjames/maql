import heapq
import numpy as np

class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def heuristic(self, a, b):
        # Manhattan distance on a square grid
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(self, node):
        dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]  # 4-way connectivity
        result = []
        for dir in dirs:
            neighbor = (node[0] + dir[0], node[1] + dir[1])
            if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols and not self.grid[neighbor[0]][neighbor[1]]:
                result.append(neighbor)
        return result

    def search(self, start, goal):
        if self.grid[start[0]][start[1]] or self.grid[goal[0]][goal[1]]:
            return None  # Start or goal is blocked

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == goal:
                break

            for next in self.neighbors(current):
                new_cost = cost_so_far[current] + 1  # Assumes a grid, cost between neighbors is 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        return self.reconstruct_path(came_from, start, goal)

    def reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

# Example usage:
grid = [[0, 0, 0, 0, 1],  # 0 represents free space, 1 represents an obstacle
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]]
astar = AStar(grid)
start = (0, 0)  # Start node
goal = (4, 4)   # Goal node
path = astar.search(start, goal)
print("Path:", path)
