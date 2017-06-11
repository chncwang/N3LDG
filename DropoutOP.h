#ifndef DropoutOP
#define DropoutOP

/*
*  DropoutOP.h:
*  drop out
*  
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

using namespace Eigen;


class DropoutNode : public Node {
public:
  PNode in;
  vector<int> ids;
  dtype dropvalue;
  bool bTrain;
public:
  DropoutNode() : Node() {
    node_type = "drop";
  }

  ~DropoutNode() {
    ids.clear();
  }

public:
  virtual inline void clearValue() {
    Node::clearValue();
    ids.clear();
    bTrain = false;
  }

  inline void setParam(dtype ratio) {
    assert(ratio >= 0 && ratio <= 1);
    dropvalue = ratio;
  }

public:
  void forward(Graph *cg, PNode x) {
    in = x;
    loss = 0;
    bTrain = cg->train;

    degree = 1;
    in->parents.push_back(this);
    cg->addNode(this);
  }

public:
  void compute() {
    if (bTrain) {
      std::vector<int> indexes;
      for (int i = 0; i < dim; ++i) {
        indexes.push_back(i);
      }

      int dropNum = (int) (dim * dropvalue);
      random_shuffle(indexes.begin(), indexes.end());

      for (int idx = 0; idx < dim; idx++) {
        int targetId = indexes[idx];
        if (idx < dropNum) {
          val[targetId] = 0;
          ids.push_back(targetId);
        } else {
          val[targetId] = in->val[targetId];
        }
      }
    } else {
      val.vec() = in->val.vec() * (1 - dropvalue);
    }
  }

  inline void backward() {
    if (!bTrain) {
      std::cout << "could not run backward if not in training" << std::endl;
      return;
    }

    Tensor1D mask;
    mask.init(dim, NULL);
    mask = 1;
    for (int idx = 0; idx < ids.size(); idx++) {
      mask[ids[idx]] = 0;
    }

    in->loss.vec() += loss.vec() * mask.vec();
  }

public:
  inline PExecute generate();

  // better to rewrite for deep understanding
  inline bool typeEqual(PNode other) {
    return Node::typeEqual(other);
  }

};


class DropoutExecute : public Execute {
public:
  inline void forward() {
    int count = batch.size();
    for (int idx = 0; idx < count; idx++) {
      DropoutNode *ptr = (DropoutNode *) batch[idx];
      ptr->compute();
    }
  }

  inline void backward() {
    int count = batch.size();
    for (int idx = 0; idx < count; idx++) {
      DropoutNode *ptr = (DropoutNode *) batch[idx];
      ptr->backward();
    }
  }

};


inline PExecute DropoutNode::generate() {
  DropoutExecute *exec = new DropoutExecute();
  exec->batch.push_back(this);
  return exec;
}


#endif
