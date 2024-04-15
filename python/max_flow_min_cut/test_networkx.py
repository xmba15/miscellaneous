import networkx as nx
from matplotlib import pyplot as plt


def main():
    dg = nx.DiGraph()
    dg.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
    nx.draw_networkx(dg)
    plt.savefig("graph.jpg")
    plt.show()


if __name__ == "__main__":
    main()
