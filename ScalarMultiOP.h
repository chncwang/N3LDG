#ifndef N3LDG_SCALAR_MULTI_OP_H
#define N3LDG_SCALAR_MULTI_OP_H

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class ScalarMultiNode : public Node {
public:
  PNode scalar = NULL;
  PNode vector = NULL;

  ScalarMultiNode() {
    node_type = "sclar_multiply";
  }

  void clearValue() override {
    scalar = NULL;
    vector = NULL;
    Node::clearValue();
  }

  void forward(Graph *graph, PNode scalar_node, PNode vector_node) {
    n3ldg_assert(scalar_node->dim == 1, "scalr_node dim should be 1 but is " + scalar_node->dim);
    scalar = scalar_node;
    vector = vector_node;
    degree = 0;
    scalar->addParent(this);
    vector->addParent(this);
    graph->addNode(this);
  }

  void compute() override {

  }

};

#endif