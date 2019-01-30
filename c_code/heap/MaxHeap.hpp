/**
 * @file    MaxHeap.hpp
 *
 * @brief   Test MaxHeap
 *
 * @author  btran
 *
 * @date    2019-01-30
 *
 * miscellaneous framework
 *
 * Copyright (c) organization
 *
 */

#include <algorithm>
#include <boost/graph/graphviz.hpp>
#include <iostream>
#include <limits>
#include <queue>
#include <utility>
#include <vector>
#include <sstream>
#include <string>

typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::directedS,
    boost::property<boost::vertex_color_t, boost::default_color_type>,
    boost::property<boost::edge_weight_t, int>>
    Graph;

typedef std::pair<int, int> Edge;

template <class T> class MaxHeap
{
public:
  static constexpr T TMAX = std::numeric_limits<T>::max();
  static constexpr T TMIN = std::numeric_limits<T>::min();

  MaxHeap() = default;
  virtual ~MaxHeap() = default;

  int parent(int i);
  int leftChild(int i);
  int rightChild(int i);

  T getMax();

  // heapify a subtree with the root at given index
  void MaxHeapify(int i);

  // remove maximum value from heap
  T extractMax();
  void increaseKey(int i, T newVal);

  void deleteKey(int i);
  void insertKey(T k);

  // height of a subtree
  int nodeHeight(int i);
  // height of the whole tree
  int treeHeight(int i);
  void BFSTraversal(int i);
  void writeGraph(std::string path);
  // get all edges of subtree
  std::vector<Edge> getEdges(int i);

  static void swap(T &x, T &y);

  std::vector<T> const &getHarr() const
  {
    return harr;
  }

protected:
  std::vector<T> harr;
};

template <class T> void MaxHeap<T>::swap(T &x, T &y)
{
  T temp = x;
  x = y;
  y = temp;
}

template <class T> int MaxHeap<T>::parent(int i)
{
  return (i - 1) / 2;
}

template <class T> int MaxHeap<T>::leftChild(int i)
{
  return 2 * i + 1;
}

template <class T> int MaxHeap<T>::rightChild(int i)
{
  return 2 * i + 2;
}

template <class T> T MaxHeap<T>::getMax()
{
  if (harr.empty()) {
    return TMAX;
  }
  return harr[0];
}

template <class T> void MaxHeap<T>::MaxHeapify(int i)
{
  int l = leftChild(i);
  int r = rightChild(i);
  int biggest = i;
  if (l < harr.size() && harr[l] > harr[i]) {
    biggest = l;
  }
  if (r < harr.size() && harr[r] > harr[i]) {
    biggest = r;
  }
  if (biggest != i) {
    swap(harr[i], harr[biggest]);
    MaxHeapify(biggest);
  }
}

template <class T> T MaxHeap<T>::extractMax()
{
  if (harr.empty()) {
    return TMAX;
  }
  T output = harr.front();
  harr.erase(harr.begin());
  MaxHeapify(0);
  return output;
}

template <class T> void MaxHeap<T>::increaseKey(int i, T newVal)
{
  harr[i] = newVal;
  while (i != 0 && harr[parent(i)] < harr[i]) {
    swap(harr[parent(i)], harr[i]);
    i = parent(i);
  }
}

template <class T> void MaxHeap<T>::deleteKey(int i)
{
  increaseKey(i, TMAX);
  extractMax();
}

template <class T> void MaxHeap<T>::insertKey(T k)
{
  int i = harr.size();
  harr.push_back(k);

  while (i != 0 && harr[parent(i)] < harr[i]) {
    swap(harr[parent(i)], harr[i]);
    i = parent(i);
  }
}

template <class T> int MaxHeap<T>::nodeHeight(int i)
{
  if (i >= harr.size()) {
    return 0;
  }
  int l = leftChild(i);
  int r = rightChild(i);
  return std::max(nodeHeight(l), nodeHeight(r)) + 1;
}

template <class T> int MaxHeap<T>::treeHeight(int i)
{
  return nodeHeight(0);
}

template <class T> void MaxHeap<T>::BFSTraversal(int i)
{
  if (i >= harr.size()) {
    return;
  }

  std::queue<int> q;
  q.push(i);
  while (!q.empty()) {
    int i = q.front();
    std::cout << harr[i] << "\n";
    q.pop();
    int l = leftChild(i);
    int r = rightChild(i);
    if (l < harr.size()) {
      q.push(l);
    }
    if (r < harr.size()) {
      q.push(r);
    }
  }
}

template <class T> std::vector<Edge> MaxHeap<T>::getEdges(int i)
{
  std::vector<Edge> edges;

  std::queue<int> nodes;
  nodes.push(i);

  while (!nodes.empty()) {
    i = nodes.front();
    nodes.pop();

    int l = leftChild(i);
    int r = rightChild(i);

    if (l < harr.size()) {
      edges.push_back(Edge(i, l));
      nodes.push(l);
    }
    if (r < harr.size()) {
      edges.push_back(Edge(i, r));
      nodes.push(r);
    }
  }

  return edges;
}

template <class T> void MaxHeap<T>::writeGraph(std::string path)
{
  std::vector<Edge> edges = getEdges(0);
  int nEdges = edges.size();

  char **names = new char*[harr.size()];
  for (int i = 0; i < harr.size(); ++i) {
    std::stringstream ss;
    ss << harr[i];
    names[i] = new char[ss.str().length() + 1];
    ss >> names[i];
  }

  T * weights = new T[nEdges];
  std::fill(weights, weights + nEdges, 1);

  Graph gWrite(edges.begin(), edges.end(), weights, harr.size());
  std::ofstream outputFile;
  outputFile.open(path);
  boost::write_graphviz(outputFile, gWrite, boost::make_label_writer(names));

  outputFile.close();
  delete[] weights;
  for (int i = 0; i < harr.size(); ++i) {
    delete[] names[i];
  }
  delete[] names;
}
