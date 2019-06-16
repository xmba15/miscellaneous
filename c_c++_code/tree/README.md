# Tree Data Structure #
## Binary Tree ##
### Definition ###
1. Depth of a node: the number of edges from the node to the tree's root node. A root node has a depth of 0.
2. Level: is defined by 1 + the number of connections between the node and the root. = depth + 1
3. Height of a node: the number of edges on the longest path from the node to a leaf. A leaf node has a height of 0.
4. Height of a tree = height of its root node = the depth of the deepest node.
5. Diameter or width of a tree = number of nodes on the longest path between any two leaf nodes.

## Binary Search Tree ##
### Definition ###
A node-based binary tree data structure with the following properties:
1. The left subtree of a node contains only nodes with keys lesser than the key of that node
2. Same for a right subtree
3. The left and right subtree must also be a binary search tree

## Red Black Tree ##
1. Every node is colored red or black
2. Root node is a black node
3. NULL children count as black nodes
4. Children of a red node are black nodes
5. For all nodes x:
   - all paths from x to NIL's have the same number of black nodes on them

## Graphviz for visualization ##
### Sample dot file ###
sample.dot

```
digraph {
    1 -> 0.5;
    10 -> 3;
    10 -> 12;
    1 -> 10;
}
```
### Commands ###
```
# dot -T[format] [dot file] > [output file name].[format]
dot -Tpng sample.dot > output.png
```
