#ifndef N3LDG_TRI_OP_H
#define N3LDG_TRI_OP_H

#include <array>
#include "MultiOP.h"

class TriParams : public MultiParams<3> {
public:
  Param& w1() {
    return _params.at(0);
  }

  Param& w2() {
    return _params.at(1);
  }

  Param& w3() {
    return _params.at(2);
  }
};

class TriNode : public MultiNode<3> {
public:
	TriNode() {
		node_type = "tri";
	}
	~TriNode() = default;
};

#endif
