#ifndef N3LDG_SCALAR_MULTI_OP_H
#define N3LDG_SCALAR_MULTI_OP_H

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

class ScalarMultiNode : public DropoutableNode {
public:
  PNode scalar = NULL;
  PNode vector = NULL;

  ScalarMultiNode() {
      node_type = toString(NodeType::SCALAR_MULTIPLY);
  }

  void clearValue() override {
    scalar = NULL;
    vector = NULL;
    Node::clearValue();
  }

  void forward(Graph *graph, PNode scalar_node, PNode vector_node) {
    n3ldg_assert(scalar_node->dim == 1, "scalr_node dim should be 1 but is " << scalar_node->dim);
    scalar = scalar_node;
    vector = vector_node;
    degree = 0;
    scalar->addParent(this);
    vector->addParent(this);
    graph->addNode(this);
  }

  void compute() override {
    val.vec() = scalar->val[0] * vector->val.vec();
  }

  void backward() override {
    scalar->loss.mat() += loss.mat() * vector->val.tmat();
    n3ldg_assert(scalar->loss.dim == 1, "scalar_node dim should be 1 bug is " + scalar->loss.dim);
    vector->loss.vec() += loss.vec() * scalar->val.vec();
  }
};

#endif