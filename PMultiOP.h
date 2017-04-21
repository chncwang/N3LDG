#ifndef PMultiOP
#define PMultiOP

/*
*  PMultiOP.h:
*  pointwise multiplication
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

struct PMultiNode : Node {
  PNode in1, in2;
public:
  PMultiNode() : Node() {
    in1 = NULL;
    in2 = NULL;
    node_type = "point-multiply";
  }
public:
  virtual inline void clearValue() {
    Node::clearValue();
    in1 = NULL;
    in2 = NULL;
  }

public:
  void forward(Graph *cg, PNode x1, PNode x2) {
    in1 = x1;
    in2 = x2;
    degree = 2;
    x1->parents.push_back(this);
    x2->parents.push_back(this);
    cg->addNode(this);
  }


public:
  // better to rewrite for deep understanding
  inline bool typeEqual(PNode other) {
    return Node::typeEqual(other);
  }

  inline PExcute generate();
};


struct PMultiExcute :Excute {
public:
  Tensor1D y, x1, x2;
  int sumDim;

public:
  inline void  forward() {
    int count = batch.size();
    sumDim = 0;
    for (int idx = 0; idx < count; idx++) {
      sumDim += batch[idx]->dim;
    }

    y.init(sumDim, NULL);
    x1.init(sumDim, NULL);
    x2.init(sumDim, NULL);

    int offset = 0;
    for (int idx = 0; idx < count; idx++) {
      PMultiNode* ptr = (PMultiNode*)batch[idx];
      for (int idy = 0; idy < ptr->dim; idy++) {
        x1[offset + idy] = ptr->in1->val[idy];
        x2[offset + idy] = ptr->in2->val[idy];
      }
      offset += ptr->dim;
    }

    y.vec() = x1.vec() * x2.vec();

    offset = 0;
    for (int idx = 0; idx < count; idx++) {
      PMultiNode* ptr = (PMultiNode*)batch[idx];
      for (int idy = 0; idy < ptr->dim; idy++) {
        ptr->val[idy] = y[offset + idy];
      }
      offset += ptr->dim;
    }

  }

  inline void  backward() {
    Tensor1D ly, lx1, lx2;
    ly.init(sumDim, NULL);
    lx1.init(sumDim, NULL);
    lx2.init(sumDim, NULL);

    int count = batch.size();
    int offset = 0;
    for (int idx = 0; idx < count; idx++) {
      PMultiNode* ptr = (PMultiNode*)batch[idx];
      for (int idy = 0; idy < ptr->dim; idy++) {
        ly[offset + idy] = ptr->loss[idy];
      }
      offset += ptr->dim;
    }

    lx1.vec() = ly.vec() * x2.vec();
    lx2.vec() = ly.vec() * x1.vec();

    offset = 0;
    for (int idx = 0; idx < count; idx++) {
      PMultiNode* ptr = (PMultiNode*)batch[idx];
      for (int idy = 0; idy < ptr->dim; idy++) {
        ptr->in1->loss[idy] += lx1[idy];
        ptr->in2->loss[idy] += lx2[idy];
      }
      offset += ptr->dim;
    }
  }

};


inline PExcute PMultiNode::generate() {
  PMultiExcute* exec = new PMultiExcute();
  exec->batch.push_back(this);
  return exec;
}


#endif
