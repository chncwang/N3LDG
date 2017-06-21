#ifndef N3LDG_BI_OP_H
#define N3LDG_BI_OP_H

#include <array>
#include "MultiOP.h"

class BiParams : public MultiParams<2> {
public:
  Param& w1() {
    return _params.at(0);
  }

  Param& w2() {
    return _params.at(1);
  }
};

class BiNode : public MultiNode<2> {
public:
	BiNode() {
		node_type = "bi";
	}
};

#endif
