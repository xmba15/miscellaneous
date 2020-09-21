#include <iostream>
#include <map>
#include <set>

struct Node
{
  explicit Node(int i) : _i(i) {}
  int _i;

  // order is important
  bool operator<(const Node& rhs) const
  {
    return _i < rhs._i;
  }
};

std::ostream &operator <<(std::ostream &os, const Node &n) {
  os << n._i;
  return os;
}

using Graph = std::multimap<Node, Node>;
// this equals to
// using Graph = std::map<Node, std::set<Node> >;

int main(int argc, char *argv[]) {
  Graph g;
  Node node1(1), node2(2), node3(3), node4(4), node5(5), node6(6), node7(7), node8(8), node9(9);

  // g.insert (Graph::value_type(node1, node3));
  g.insert (std::make_pair(node1, node3));
  g.insert (Graph::value_type(node1, node4));
  g.insert (Graph::value_type(node1, node5));
  g.insert (Graph::value_type(node2, node6));
  g.insert (Graph::value_type(node3, node6));
  g.insert (Graph::value_type(node4, node7));
  g.insert (Graph::value_type(node5, node7));
  g.insert (Graph::value_type(node5, node8));
  g.insert (Graph::value_type(node5, node9));
  g.insert (Graph::value_type(node9, node5));

  for (auto it = g.begin(); it != g.end(); it = g.upper_bound(it->first)) {
    std::cout <<  it->first._i << ":";

    // the range of elements with key equal to it->first
    std::pair<Graph::iterator, Graph::iterator> valsOfKeyIt = g.equal_range(it->first);
    for (auto it = valsOfKeyIt.first; it != valsOfKeyIt.second; ++it) {
      std::cout << it->second << " ";
    }
    std::cout << "\n";
  }

  return 0;
}
