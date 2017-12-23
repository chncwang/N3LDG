/*
 * BaseParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef BasePARAM_H_
#define BasePARAM_H_

#include "MyTensor.h"

#if USE_GPU
#include "n3ldg_cuda.h"
#endif

struct BaseParam {
#if USE_GPU
    n3ldg_cuda::Tensor2D val;
    n3ldg_cuda::Tensor2D grad;
#else
    Tensor2D val;
    Tensor2D grad;
#endif
  public:
    virtual inline void initial(int outDim, int inDim) = 0;
    virtual inline void updateAdagrad(dtype alpha, dtype reg, dtype eps) = 0;
    virtual inline void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) = 0;
    virtual inline int outDim() = 0;
    virtual inline int inDim() = 0;
    virtual inline void clearGrad() = 0;

    // Choose one point randomly
    virtual inline void randpoint(int& idx, int &idy) = 0;
    virtual inline dtype squareGradNorm() = 0;
    virtual inline void rescaleGrad(dtype scale) = 0;
    virtual inline void save(std::ofstream &os)const = 0;
    virtual inline void load(std::ifstream &is) = 0;
};

#endif /* BasePARAM_H_ */
