#ifndef CONCAT
#define CONCAT

/*
*  Concat.h:
*  concatenatation operation.
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/


#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

struct ConcatNode : Node{
public:
	int nSize;
	vector<int> inDims;
	vector<PNode> ins;

public:
	ConcatNode() : Node(){
		nSize = 0;
		inDims.clear();
		ins.clear();
    node_type = "concat";
	}

	inline void clearValue(){
		Node::clearValue();
	}
	
	inline void init(int dim, AlignedMemoryPool* mem = NULL){
		Node::init(dim, mem);
	}

public:
	void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0){
			std::cout << "empty inputs for concat" << std::endl;
			return;
		}

		ins.clear();
		for (int i = 0; i < x.size(); i++){
			ins.push_back(x[i]);
		}

    degree = ins.size();
    for (int i = 0; i < degree; ++i) {
      ins[i]->parents.push_back(this);
    }

		cg->addNode(this);
	}


	void forward(Graph *cg, PNode x1, PNode x2){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);

    degree = ins.size();
    for (int i = 0; i < degree; ++i) {
      ins[i]->parents.push_back(this);
    }

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);

    degree = ins.size();
    for (int i = 0; i < degree; ++i) {
      ins[i]->parents.push_back(this);
    }

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);

    degree = ins.size();
    for (int i = 0; i < degree; ++i) {
      ins[i]->parents.push_back(this);
    }

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);

    degree = ins.size();
    for (int i = 0; i < degree; ++i) {
      ins[i]->parents.push_back(this);
    }

		cg->addNode(this);
	}

	void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6){
		ins.clear();
		ins.push_back(x1);
		ins.push_back(x2);
		ins.push_back(x3);
		ins.push_back(x4);
		ins.push_back(x5);
		ins.push_back(x6);

    degree = ins.size();
    for (int i = 0; i < degree; ++i) {
      ins[i]->parents.push_back(this);
    }

		cg->addNode(this);
	}



 public:
   inline PExcute generate();

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
      return Node::typeEqual(other);
    }

public:
	inline void compute(){
		nSize = ins.size();
		inDims.clear();
		int curDim = 0;
		for (int i = 0; i < nSize; ++i){
			inDims.push_back(ins[i]->val.dim);
			curDim += inDims[i];
		}
		if(curDim != dim){
			std::cout << "input dim size not match" << std::endl;
			return;
		}

		int offset = 0;
		for (int i = 0; i < nSize; ++i){
			for (int idx = 0; idx < inDims[i]; idx++){
				val[offset + idx] = ins[i]->val[idx];
			}
			offset += inDims[i];
		}
	}


  void backward() {
    int offset = 0;
    for (int i = 0; i < nSize; ++i) {
      for (int idx = 0; idx < inDims[i]; idx++) {
        ins[i]->loss[idx] += loss[offset + idx];
      }
      offset += inDims[i];
    }
  }

};


struct ConcatExcute :Excute {
public:
  inline void  forward() {
    int count = batch.size();

    for (int idx = 0; idx < count; idx++) {
      ConcatNode* ptr = (ConcatNode*)batch[idx];
      ptr->compute();
    }
  }

  inline void backward() {
    int count = batch.size();
    for (int idx = 0; idx < count; idx++) {
      ConcatNode* ptr = (ConcatNode*)batch[idx];
      ptr->backward();
    }
  }
};


inline PExcute ConcatNode::generate() {
  ConcatExcute* exec = new ConcatExcute();
  exec->batch.push_back(this);
  return exec;
}

#endif
