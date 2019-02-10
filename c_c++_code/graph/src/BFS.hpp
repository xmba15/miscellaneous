/**
 * @file    BFS.hpp
 *
 * @brief   test BFS graph traversal
 *
 * @author  xmba15
 *
 * @date    2019-01-29
 *
 * miscellaneous
 *
 * Copyright (c) organization
 *
 */

#ifndef BFS_HPP_
#define BFS_HPP_

#include <list>
#include <queue>
#include <iostream>

class Graph
{
public:
  Graph(int V);
  virtual ~Graph() = default;
  void addEdge(int v, int w);
  void BFS(int s);
private:
  int V;
  std::list<int> *adj;
};

Graph::Graph(int V) : V(V)
{
  adj =  new std::list<int>[V];
}

void Graph::addEdge(int v, int w)
{
  adj[v].push_back(w);
}

void Graph::BFS(int s) {
  bool *visited = new bool[V];
  for (int i = 0; i < V; ++i) {
    visited[i] = false;
  }

  std::list<int> queue;
  visited[s] = true;
  queue.push_back(s);

  std::list<int>::iterator i;

  while (!queue.empty()) {
    s = queue.front();
    std::cout << s << "\n";
    queue.pop_front();

    for (i = adj[s].begin(); i != adj[s].end(); ++i) {
      if (!visited[*i])
      {
        visited[*i] = true;
        queue.push_back(*i);
      }
    }
  }
}


#endif /* BFS_HPP_ */
