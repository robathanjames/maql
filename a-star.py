import heapq

class Graph:
    def __init__(self):
        self.edges = {}

    def neighbors(self, node):
        return self.edges[node]

    def cost(self, from_node, to_node):
        return self.edges[from_node][to_node]

class AStar:
    def __init__(self, graph):
        self.graph = graph

    def heuristic(self, a, b):
        # Simple heuristic: Euclidean distance (replace with appropriate heuristic for your use case)
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def search(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == goal:
                break

            for next in self.graph.neighbors(current):
                new_cost = cost_so_far[current] + self.graph.cost(current, next)
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
graph = Graph()
graph.edges = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 2, 'E': 5},
    'C': {'A': 4, 'F': 5},
    'D': {'B': 2, 'G': 3},
    'E': {'B': 5, 'H': 4},
    'F': {'C': 5, 'I': 6},
    'G': {'D': 3},
    'H': {'E': 4},
    'I': {'F': 6}
}

astar = AStar(graph)
start = 'A'  # Start node
goal = 'I'   # Goal node
path = astar.search(start, goal)
print("Path:", path)
