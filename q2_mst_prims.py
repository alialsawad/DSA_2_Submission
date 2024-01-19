from math import ceil
import networkx as nx
import matplotlib.pyplot as plt
from heapq import heappush, heappop


class PrimsAlgorithm:

    def __init__(self, G):
        self.graph = G

    def run(self, starting_node):
        print(
            f"Starting Prim's Algorithm on the following graph:\n{G.edges(data=True)}")

        visited = set()
        min_heap = [(0, None, starting_node)]
        mst_edges = []
        total_weight = 0

        while len(visited) < len(G):
            w, u_prev, u = heappop(min_heap)

            if u in visited:
                continue

            if u_prev is not None:
                mst_edges.append((u_prev, u, w))

            visited.add(u)
            total_weight += w

            for v, data in G[u].items():
                if v not in visited:
                    heappush(min_heap, (data['weight'], u, v))

        print(f"Minimum Spanning Tree Edges: {mst_edges}")
        print(f"Total Weight: {total_weight}")
        return mst_edges, total_weight


class MstVisualizer:

    def __init__(self, G, mst_edges, total_weight, layout=None):
        self.graph = G
        self.mst_edges = mst_edges
        self.total_weight = total_weight
        self.pos = nx.shell_layout(G) if layout is None else layout

    def draw_graph(self):
        # +1 for original graph, +1 for mst
        num_steps = len(self.mst_edges) + 2
        num_cols = 3
        num_rows = ceil(num_steps / num_cols)

        # Create a figure with a subplot for each step of the algorithm
        _, axes = plt.subplots(nrows=num_rows, ncols=num_cols,
                               figsize=(20, 5))
        axes = axes.flatten()



        # A list of highlighted edges as they were added to the MST
        cur_mst_edges = set()
        for i in range(num_steps):
            if i > 0 and i <= len(self.mst_edges):
                cur_mst_edge = self.mst_edges[i - 1]
                cur_mst_edges.add((cur_mst_edge[0], cur_mst_edge[1]))
            # Draw the current graph representation at the corresponding axis
            self._draw_graph_step(axes[i], i, cur_mst_edges)
            
        # remove extra axes (subplots) that we don't need
        for i in range(num_steps, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    def _draw_graph_step(self, ax, step, cur_mst_edges):
        not_mst = step <= len(self.mst_edges)

        graph_title = ""
        if not_mst:
            graph_title = "Original Graph" if step == 0 else f"Step {step}: Adding Edge {self.mst_edges[step - 1]}"
        else:
            graph_title = f"Minimum Spanning Tree. Total Weight: {self.total_weight}"

        edge_labels = nx.get_edge_attributes(self.graph, 'weight') if not_mst else {
            (edge[0], edge[1]): edge[2] for edge in self.mst_edges}

        graph_info = {
            "mst_edges": cur_mst_edges,
            "edge_labels": edge_labels,
            "title": graph_title,
        }
        if not_mst:
            # Draw both the edges of the original graph and the edges of the MST
            graph_info["edges"] = self.graph.edges

        self._draw_graph_info(ax, graph_info)
    
    def _draw_graph_info(self, ax, graph_info):
        mst_edges = graph_info.get("mst_edges", [])
        # Remove duplicate edges so that we don't draw them twice (better visualization)
        graph_edges = list(filter(
            lambda edge: edge not in mst_edges and
            (edge[1], edge[0]) not in mst_edges, graph_info.get("edges", [])))

        all_edges = list(graph_edges) + list(mst_edges)

        edge_colors = ['black'] * len(graph_edges) + ['red'] * len(mst_edges)

        nx.draw_networkx(self.graph, self.pos, ax=ax, node_color='lightblue',
                         node_size=600, edgelist=all_edges, edge_color=edge_colors, width=1.25)
        nx.draw_networkx_edge_labels(
            self.graph, self.pos, edge_labels=graph_info.get("edge_labels"), ax=ax, font_size=8)
        ax.set_title(
            graph_info.get("title"), fontsize=12)


def create_graph(edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G


edges = [("A", "C", 16), ("A", "B", 4), ("A", "D", 3), ("A", "E", 1),
         ("E", "D", 1), ("C", "B", 5), ("C", "F", 5), ("B", "F", 9),
         ("B", "D", 8), ("D", "F", 12), ("F", "Z", 1)]

G = create_graph(edges)

prims_algorithm = PrimsAlgorithm(G)
mst_edges, total_weight = prims_algorithm.run(starting_node="A")

mst_visualizer = MstVisualizer(G, mst_edges, total_weight)
mst_visualizer.draw_graph()
