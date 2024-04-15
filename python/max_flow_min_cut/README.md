# 📝 max flow min cut
***

### Algorithm: Ford-Fulkerson

Inputs: Given network G=(V,E) with flow capacity c, a source node s, and a sink node t

Output: Compute max flow from s to t

1. Initialize f(u,v) ← 0 for all edges (u, v)
2. While there is an augmenting path p from s to t in residual network G_f, such that residual capacity c_f(u,v)=c(u,v)-f(u,v)>0 for all edges (u,v)∈p:
   1. Find c_f(p)=min{c_f(u,v): (u,v)∈p}
   2. For each edge (u,v)∈
      1. f(u,v) ← f(u,v)+c_f(p)
      1. f(v,u) ← f(v,u)-c_f(p)


### Algorithm: Minium cut

1. Run Ford-Fulkerson algorithm to obtain the final residual graph
2. Find the set of vertices that are reachable from the source in the residual graph
3. All edges which are from a reachable vertex to a non-reachable vertex are minimum cut edges

## 🎛  Dependencies
***

## 🔨 How to Build ##
mamba env create --file environment.yml
mamba activate graph_theory
***

## :running: How to Run ##
***

## :gem: References ##
***
