#ifndef POOLING
#define POOLING

/*
*  Pooling.h:
*  pool operation, max, min, average and sum pooling
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"


class PoolNode : public Node{
public:
  vector<Tensor1D> masks;
  vector<PNode> ins;
  int nSize;

public:
  PoolNode() : Node() {
    ins.clear();
  }

  ~PoolNode() {
    masks.clear();
    ins.clear();
  }

  inline void clearValue() {
    Node::clearValue();
    ins.clear();
  }

  inline void setParam(int maxsize) {
    masks.resize(maxsize);
  }

  inline void init(int dim, AlignedMemoryPool* mem = NULL) {
    Node::init(dim, mem);
    int count = masks.size();
    for (int idx = 0; idx < count; idx++) {
      masks[idx].init(dim, mem);
    }
  }

public:
  virtual void forward(Graph *cg, const vector<PNode>& x) = 0;


public:
  inline PExecute generate();

  // better to rewrite for deep understanding
  inline bool typeEqual(PNode other) {
    return Node::typeEqual(other);
  }

public:
  void compute() {
    val.zero();
    for (int i = 0; i < nSize; ++i) {
      val.vec() += masks[i].vec() * ins[i]->val.vec();
    }
  }

  void backward() {
    for (int i = 0; i < nSize; i++) {
      ins[i]->loss.vec() += loss.vec() * masks[i].vec();
    }
  }

};

class MaxPoolNode : public PoolNode {
public:
  MaxPoolNode() : PoolNode() {
    node_type = "max-pooling";
  }

public:
  //Be careful that the row is the dim of input vector, and the col is the number of input vectors
  //Another point is that we change the input vectors directly.
  void forward(Graph *cg, const vector<PNode>& x) {
    if (x.size() == 0) {
      std::cout << "empty inputs for max pooling" << std::endl;
      return;
    }
    nSize = x.size();
    ins.clear();
    for (int i = 0; i < nSize; i++) {
      ins.push_back(x[i]);
    }

    for (int i = 0; i < nSize; ++i) {
      if (ins[i]->val.dim != dim) {
        std::cout << "input matrixes are not matched" << std::endl;
        clearValue();
        return;
      }
      masks[i].zero();
    }

    for (int idx = 0; idx < dim; idx++) {
      int maxIndex = -1;
      for (int i = 0; i < nSize; ++i) {
        if (maxIndex == -1 || ins[i]->val[idx] > ins[maxIndex]->val[idx]) {
          maxIndex = i;
        }
      }
      masks[maxIndex][idx] = 1.0;
    }

    degree = nSize;
    for (int i = 0; i < nSize; i++) {
      ins[i]->parents.push_back(this);
    }

    cg->addNode(this);
  }

};


class SumPoolNode : public PoolNode {
public:
  SumPoolNode() : PoolNode() {
    node_type = "sum-pooling";
  }

public:
  //Be careful that the row is the dim of input vector, and the col is the number of input vectors
  //Another point is that we change the input vectors directly.
  void forward(Graph *cg, const vector<PNode>& x) {
    if (x.size() == 0) {
      std::cout << "empty inputs for max pooling" << std::endl;
      return;
    }

    nSize = x.size();
    ins.clear();
    for (int i = 0; i < nSize; i++) {
      ins.push_back(x[i]);
    }

    for (int i = 0; i < nSize; ++i) {
      if (ins[i]->val.dim != dim) {
        std::cout << "input matrixes are not matched" << std::endl;
        clearValue();
        return;
      }
      masks[i] = 1.0;
    }

    degree = nSize;
    for (int i = 0; i < nSize; i++) {
      ins[i]->parents.push_back(this);
    }

    cg->addNode(this);
  }

};


class MinPoolNode : public PoolNode {
public:
  MinPoolNode() : PoolNode() {
    node_type = "min-pooling";
  }

public:
  //Be careful that the row is the dim of input vector, and the col is the number of input vectors
  //Another point is that we change the input vectors directly.
  void forward(Graph *cg, const vector<PNode>& x) {
    if (x.size() == 0) {
      std::cout << "empty inputs for max pooling" << std::endl;
      return;
    }
    nSize = x.size();
    ins.clear();
    for (int i = 0; i < nSize; i++) {
      ins.push_back(x[i]);
    }

    for (int i = 0; i < nSize; ++i) {
      if (ins[i]->val.dim != dim) {
        std::cout << "input matrixes are not matched" << std::endl;
        clearValue();
        return;
      }
      masks[i].zero();
    }


    for (int idx = 0; idx < dim; idx++) {
      int minIndex = -1;
      for (int i = 0; i < nSize; ++i) {
        if (minIndex == -1 || ins[i]->val[idx] < ins[minIndex]->val[idx]) {
          minIndex = i;
        }
      }
      masks[minIndex][idx] = 1.0;
    }

    degree = nSize;
    for (int i = 0; i < nSize; i++) {
      ins[i]->parents.push_back(this);
    }

    cg->addNode(this);
  }

};



class AvgPoolNode : public PoolNode {
public:
  AvgPoolNode() : PoolNode() {
    node_type = "avg-pooling";
  }

public:
  //Be careful that the row is the dim of input vector, and the col is the number of input vectors
  //Another point is that we change the input vectors directly.
  void forward(Graph *cg, const vector<PNode>& x) {
    if (x.size() == 0) {
      std::cout << "empty inputs for avg pooling" << std::endl;
      return;
    }

    nSize = x.size();
    ins.clear();
    for (int i = 0; i < nSize; i++) {
      ins.push_back(x[i]);
    }

    for (int i = 0; i < nSize; ++i) {
      if (ins[i]->val.dim != dim) {
        std::cout << "input matrixes are not matched" << std::endl;
        clearValue();
        return;
      }
      masks[i] = 1.0 / nSize;
    }

    degree = nSize;
    for (int i = 0; i < nSize; i++) {
      ins[i]->parents.push_back(this);
    }

    cg->addNode(this);
  }
};


class PoolExecute : public Execute {

public:
  inline void  forward() {
    int count = batch.size();

    for (int idx = 0; idx < count; idx++) {
      PoolNode* ptr = (PoolNode*)batch[idx];
      ptr->compute();
    }
  }

  inline void backward() {
    int count = batch.size();
    for (int idx = 0; idx < count; idx++) {
      PoolNode* ptr = (PoolNode*)batch[idx];
      ptr->backward();
    }
  }
};

inline PExecute PoolNode::generate() {
  PoolExecute* exec = new PoolExecute();
  exec->batch.push_back(this);
  return exec;
}

#endif
