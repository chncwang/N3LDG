#ifndef N3LDG_BI_OP_H
#define N3LDG_BI_OP_H

#include <array>
#include "MultiOP.h"

class BiParams : public MultiParams<2> {
public:
  void initial(int oSize, int iSize1, int iSize2, bool isBValid = true,
      AlignedMemoryPool *pool = NULL);

  Param& w1() {
    return _params.at(0);
  }

  Param& w2() {
    return _params.at(1);
  }
};

void BiParams::initial(int oSize, int iSize1, int iSize2, bool isBValid,
    AlignedMemoryPool *pool) {
  std::array<int, 2> sizes = {iSize1, iSize2};
  MultiParams<2>::initial(oSize, sizes, isBValid, pool);
}

#endif
