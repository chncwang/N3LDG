#ifndef BucketOP
#define BucketOP

/*
*  BucketOP.h:
*  a bucket operation, for padding mainly
*  usually an inputleaf node, degree = 0
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

using namespace Eigen;



struct BucketNode : Node {
public:
  BucketNode() : Node(){
    node_type = "bucket";
	}
public:
	virtual inline void clearValue(){
		Node::clearValue();
	}

public:
	void forward(Graph *cg, dtype value) {
    val = value;
    loss = 0;
		cg->addNode(this);
	}


public:
  inline PExcute generate();

  // better to rewrite for deep understanding
  inline bool typeEqual(PNode other) {
    return Node::typeEqual(other);
  }

};




struct BucketExcute : Excute {
public:
  inline void  forward() {

  }

  inline void backward() {

  }

};


inline PExcute BucketNode::generate() {
  BucketExcute* exec = new BucketExcute();
  exec->batch.push_back(this);
  return exec;
}



#endif
