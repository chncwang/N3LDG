#ifndef BasicNode
#define BasicNode

/*
*  Node.h:
*  basic processing unit in a neural network
*  (1) we have a node structure to build user graph
*  (2) we have a execute structure to merge similar nodes that can be execute together
*  The real forward and backward are defined in Execute.
*  Every operation should define a node class and a execute class together.
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/

#include "MyTensor.h"


class Execute;

// one Node means a vector
// the col should be 1, because we aimed for NLP only
class Node {
public:
  vector<Node*> parents;
public:
	Tensor1D val;
	Tensor1D loss;
public:
	int dim;
	int degree; //means indegree
  string node_type;

	
public:
	Node(){
		dim = 0;
    degree = 0;
    parents.clear();
    node_type = "interface";
	}

  virtual ~Node() {
    val = 0;
    loss = 0;
    degree = 0;
    parents.clear();
    node_type.clear();
  }


public:
	virtual inline void clearValue(){
		val = 0;
		loss = 0;
    degree = 0;
    parents.clear();
	}
	
	virtual inline void init(int dim, AlignedMemoryPool* mem = NULL){
		this->dim = dim;
		val.init(dim, mem);
		loss.init(dim, mem);	
    parents.clear();
	}

public:

  virtual inline Execute* generate() = 0;

  virtual inline bool typeEqual(Node* other) {
    if (node_type.compare(other->node_type) == 0) {
      return true;
    }
    return false;
  }


};

typedef  Node* PNode;


class Execute {
public:
  vector<PNode> batch;

public:
  virtual ~Execute() {
    batch.clear();
  }

public:
  virtual inline void forward() = 0;
  virtual inline void backward() = 0;


  inline bool addNode(PNode in) {
    if (batch.empty()) {
      std::cout << "empty batch, strange...." << std::endl;
      return false;
    }

    if (batch[0]->typeEqual(in)) {
      batch.push_back(in);
      return true;
    }

    return false;
  }
};


typedef  Execute* PExecute;

#endif
