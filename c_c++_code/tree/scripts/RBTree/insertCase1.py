#!/usr/bin/env python
# -*- coding: utf-8 -*-
from graphviz import Digraph


def main(filename="case4_outersubtree"):
    g = Digraph(filename, format="png")

    g.node("G", color="black")
    g.node("P", color="red")

    # g.node("1", color="black", shape="triangle")
    g.node("3", color="black", shape="triangle")
    # g.node("4", color="black", shape="triangle")
    # g.node("5", color="black", shape="triangle")

    g.node("N", color="red")

    g.edges([("G", "N")])

    # g.edge("G", "N")
    g.edge("N", "P")
    # g.edge("P", "1")
    # g.edge("N", "3")

    # g.edge("G", "U")

    g.render()


if __name__ == '__main__':
    main()
