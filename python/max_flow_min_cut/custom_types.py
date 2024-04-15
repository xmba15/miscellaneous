import copy
from collections import defaultdict
from typing import List, Tuple

import networkx as nx


class DiGraph:
    def __init__(
        self,
        num_nodes: int,
    ):
        self._num_nodes = num_nodes
        self._residual_graph = defaultdict(dict)

    def add_edge(self, u, v, capacity):
        self._residual_graph[u][v] = {
            "capacity": capacity,
            "flow": 0,
        }
        self._residual_graph[v][u] = {
            "capacity": 0,
            "flow": 0,
        }

    def to_networkx_graph(self):
        dg = nx.DiGraph()
        for u in self._residual_graph:
            for v in self._residual_graph[u]:
                capacity = self._residual_graph[u][v]["capacity"]
                if capacity <= 0:
                    continue
                dg.add_edge(u, v, capacity=capacity)
        return dg

    def _dfs(
        self,
        s,
        t,
        visited: List[bool],
        path: List[Tuple],
        residual_graph: defaultdict,
    ):
        if s == t:
            return path
        visited[s] = True
        for target_node in residual_graph[s]:
            if visited[target_node]:
                continue

            residual_capacity = (
                residual_graph[s][target_node]["capacity"]
                - residual_graph[s][target_node]["flow"]
            )
            if residual_capacity <= 0:
                continue

            augmented_path = self._dfs(
                target_node,
                t,
                visited,
                path + [(s, target_node)],
                residual_graph,
            )
            if augmented_path is not None:
                return augmented_path

        return None

    def _bfs(
        self,
        s,
        t,
        visited: List[bool],
        path: List[Tuple],
        residual_graph: defaultdict,
    ):
        parents = [-1] * self._num_nodes
        visited[s] = True

        queue = [s]
        while len(queue) > 0:
            u = queue.pop(0)
            for target_node in residual_graph[u]:
                if visited[target_node]:
                    continue

                residual_capacity = (
                    residual_graph[u][target_node]["capacity"]
                    - residual_graph[u][target_node]["flow"]
                )
                if residual_capacity <= 0:
                    continue

                parents[target_node] = u
                if u == t:
                    break
                queue.append(target_node)
                visited[target_node] = True

        if parents[t] == -1:
            return None

        u = t
        while u != s:
            path.append((parents[u], u))
            u = parents[u]
        return path

    def estimate_max_flow(
        self,
        s,
        t,
        use_bfs: bool = True,
    ):
        max_flow = 0
        residual_graph = copy.deepcopy(self._residual_graph)

        while True:
            if use_bfs:
                augmented_path = self._bfs(
                    s,
                    t,
                    [False] * self._num_nodes,
                    [],
                    residual_graph,
                )
            else:
                augmented_path = self._dfs(
                    s,
                    t,
                    [False] * self._num_nodes,
                    [],
                    residual_graph,
                )

            if augmented_path is None:
                break
            flow = min(
                [
                    residual_graph[u][v]["capacity"] - residual_graph[u][v]["flow"]
                    for u, v in augmented_path
                ]
            )
            max_flow += flow
            for u, v in augmented_path:
                residual_graph[u][v]["flow"] += flow
                residual_graph[v][u]["flow"] -= flow

        return max_flow

    def estimate_min_cut(
        self,
        s,
        t,
        use_bfs: bool = True,
    ):
        residual_graph = copy.deepcopy(self._residual_graph)

        while True:
            if use_bfs:
                augmented_path = self._bfs(
                    s,
                    t,
                    [False] * self._num_nodes,
                    [],
                    residual_graph,
                )
            else:
                augmented_path = self._dfs(
                    s,
                    t,
                    [False] * self._num_nodes,
                    [],
                    residual_graph,
                )

            if augmented_path is None:
                break
            flow = min(
                [
                    residual_graph[u][v]["capacity"] - residual_graph[u][v]["flow"]
                    for u, v in augmented_path
                ]
            )
            for u, v in augmented_path:
                residual_graph[u][v]["flow"] += flow
                residual_graph[v][u]["flow"] -= flow

        visited = [False] * self._num_nodes
        self._get_reachable_from_node(residual_graph, s, visited)

        min_cut_edges = []
        for s_node in self._residual_graph:
            for target_node in self._residual_graph[s_node]:
                if (
                    self._residual_graph[s_node][target_node]["capacity"] > 0
                    and visited[s_node]
                    and not visited[target_node]
                ):
                    min_cut_edges.append((s_node, target_node))

        return min_cut_edges

    def _get_reachable_from_node(
        self,
        residual_graph,
        s_node: int,
        visited: List[bool],
    ):
        visited[s_node] = True
        for target_node in residual_graph[s_node]:
            if visited[target_node]:
                continue

            residual_capacity = (
                residual_graph[s_node][target_node]["capacity"]
                - residual_graph[s_node][target_node]["flow"]
            )
            if residual_capacity <= 0:
                continue

            self._get_reachable_from_node(
                residual_graph,
                target_node,
                visited,
            )

    def __str__(self):
        msg = ""
        for source_node in self._residual_graph:
            for target_node in self._residual_graph[source_node]:
                capacity = self._residual_graph[source_node][target_node]["capacity"]
                if capacity == 0:
                    continue
                msg += f"{source_node}->{target_node}: {capacity}\n"
        return msg


def main():
    num_nodes = 6
    di_graph = DiGraph(num_nodes)
    di_graph.add_edge(0, 1, 16)
    di_graph.add_edge(0, 2, 13)
    di_graph.add_edge(1, 2, 10)
    di_graph.add_edge(1, 3, 12)
    di_graph.add_edge(2, 1, 4)
    di_graph.add_edge(2, 4, 14)
    di_graph.add_edge(3, 2, 9)
    di_graph.add_edge(3, 5, 20)
    di_graph.add_edge(4, 3, 7)
    di_graph.add_edge(4, 5, 4)

    print(di_graph)
    print(di_graph.estimate_max_flow(0, 5))

    min_cut_edges = di_graph.estimate_min_cut(0, 5)
    print("min cut edges: ", min_cut_edges)

    dg = di_graph.to_networkx_graph()
    print(nx.maximum_flow(dg, 0, 5))
    cut_value, partition = nx.minimum_cut(dg, 0, 5)
    reachable, non_reachable = partition
    print(
        f"min cut: {cut_value}, reachable: {reachable}, non-reachable: {non_reachable}"
    )

    pos = nx.spring_layout(dg, seed=7)
    nx.draw_networkx_edge_labels(
        dg,
        pos,
        edge_labels={(i, j): w["capacity"] for i, j, w in dg.edges(data=True)},
    )
    nx.draw_networkx(
        dg,
        pos,
        with_labels=True,
        alpha=0.5,
    )
    from matplotlib import pyplot as plt

    plt.savefig("graph.jpg")
    plt.show()


if __name__ == "__main__":
    main()
