/*
 * BaseParam.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef BasePARAM_H_
#define BasePARAM_H_

#include "MyTensor.h"

struct BaseParam {
	Tensor2D val;
	Tensor2D grad;
public:
	virtual void initial(int outDim, int inDim, AlignedMemoryPool* mem) = 0;
	virtual void updateAdagrad(dtype alpha, dtype reg, dtype eps) = 0;
	virtual void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) = 0;
	virtual int outDim() = 0;
	virtual int inDim() = 0;
	virtual void clearGrad() = 0;

	// Choose one point randomly
	virtual void randpoint(int& idx, int &idy) const = 0;
	virtual dtype squareGradNorm() = 0;
	virtual void rescaleGrad(dtype scale) = 0;
	virtual void save(std::ofstream &os)const = 0;
	virtual void load(std::ifstream &is, AlignedMemoryPool* mem = NULL) = 0;
};

#endif /* BasePARAM_H_ */
