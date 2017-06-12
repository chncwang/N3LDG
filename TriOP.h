#ifndef N3LDG_TRI_OP_H
#define N3LDG_TRI_OP_H

#include <array>
#include "MultiOP.h"

class TriParams : public MultiParams<3> {
public:
  void initial(int oSize, int iSize1, int iSize2, int iSize3,
      bool isBValid = true,
      AlignedMemoryPool *pool = NULL);

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

void TriParams::initial(int oSize, int iSize1, int iSize2, int iSize3,
    bool isBValid, AlignedMemoryPool *pool) {
  std::array<int, 3> sizes = {iSize1, iSize2, iSize3};
  MultiParams<3>::initial(oSize, sizes, isBValid, pool);
}

#endif
