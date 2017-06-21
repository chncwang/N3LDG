#ifndef N3LDG_MULTI_PARAMS_H
#define N3LDG_MULTI_PARAMS_H

#include <array>
#include <functional>
#include "Param.h"
#include "ModelUpdate.h"
#include "MyLib.h"

template<int PARAM_COUNT>
class MultiParams {
  public:
    MultiParams() : _isBValid(true) {}
    virtual ~MultiParams() = default;

    void initial(int outDim, const std::array<int, PARAM_COUNT> &inDimArray,
        bool isBValid = true,
        AlignedMemoryPool* pool = NULL);

    void exportToAdaParams(ModelUpdate &ada);
    void save(std::ofstream &os) const;
    void load(std::ifstream &is, AlignedMemoryPool *pool = NULL);

    void foward(Graph *graph, const std::array<PNode, PARAM_COUNT> &node_ptrs);

    Param& w(int paramIndex) {
      return _params.at(paramIndex);
    }

    Param& b() {
      return _b;
    }

    bool isBValid() const {
      return _isBValid;
    }

  protected:
    std::array<Param, PARAM_COUNT> _params;
    Param _b;
    bool _isBValid;
};

template<int PARAM_COUNT>
void MultiParams<PARAM_COUNT>::initial(int outDim,
    const std::array<int, PARAM_COUNT> &inDimArray,
    bool isBValid,
    AlignedMemoryPool* pool) {
  for (int i = 0; i < PARAM_COUNT; ++i) {
    _params.at(i).initial(outDim, inDimArray.at(i), pool);
  }
  if (isBValid) {
    InitAsVector(_b, outDim, pool);
  }
}

template<int PARAM_COUNT>
void MultiParams<PARAM_COUNT>::exportToAdaParams(ModelUpdate &ada) {
  for (Param &param : _params) {
    ada.addParam(&param);
  }
  if (_isBValid) {
    ada.addParam(&_b);
  }
}

template<int PARAM_COUNT>
void MultiParams<PARAM_COUNT>::save(std::ofstream &os) const {
  os << _isBValid << std::endl;
  for (Param *param : _params) {
    param->save(os);
  }
  if (_isBValid) {
    _b.save(os);
  }
}

template<int PARAM_COUNT>
void MultiParams<PARAM_COUNT>::load(std::ifstream &is,
    AlignedMemoryPool *pool) {
  is >> _isBValid;
  for (Param *param : _params) {
    param->load(is, pool);
  }
  if (_isBValid) {
    _b.load(is, pool);
  }
}

template<int PARAM_COUNT>
class MultiNode : public Node {
  public:
    MultiNode();
	virtual ~MultiNode() {
		clearValue();
	}

    void init(int dim, AlignedMemoryPool *pool = NULL);
    void setParams(MultiParams<PARAM_COUNT> *params) {
      _params = params;
    }
    void clearValue() override;
    void setActivationAndDerivation(
        const std::function<dtype(dtype)> &activate,
        const std::function<dtype(dtype, dtype)> &derivate);
    void compute();
    void forward(Graph *graph, const std::array<PNode, PARAM_COUNT> &node_ptrs);
    void backward();
    PExecute generate() override;

    PNode _input_node_ptrs[PARAM_COUNT];
    MultiParams<PARAM_COUNT>* _params;
    std::function<dtype(dtype)> _activate;
    std::function<dtype(dtype, dtype)> _derivate;
    Tensor1D ty;
    Tensor1D lty;
};

template<int PARAM_COUNT>
MultiNode<PARAM_COUNT>::MultiNode() : _params(NULL) {
  for (PNode &node_ptr : _input_node_ptrs) {
    node_ptr = NULL;
  }

  _activate = ftanh;
  _derivate = dtanh;
}

template<int PARAM_COUNT>
void MultiNode<PARAM_COUNT>::init(int dim, AlignedMemoryPool *pool) {
  Node::init(dim, pool);
  ty.init(dim, pool);
  lty.init(dim, pool);
}

template<int PARAM_COUNT>
void MultiNode<PARAM_COUNT>::clearValue() {
  for (PNode &node_ptr : _input_node_ptrs) {
    node_ptr = NULL;
  }

  Node::clearValue();
  ty.zero();
  lty.zero();
}

template<int PARAM_COUNT>
void MultiNode<PARAM_COUNT>::setActivationAndDerivation(
    const std::function<dtype(dtype)> &activate,
    const std::function<dtype(dtype, dtype)> &derivate) {
  _activate = activate;
  _derivate = derivate;
}

template<int PARAM_COUNT>
void MultiNode<PARAM_COUNT>::compute() {
  ty.mat().setZero();
  for (int i = 0; i < PARAM_COUNT; ++i) {
	  Eigen::Product<Mat, Mat, 0> t = _params->w(i).val.mat() * _input_node_ptrs[i]->val.mat();
	  ty.mat() += t;
    //ty.mat() += _params->w(i).val.mat() * _input_node_ptrs[i]->val.mat();
  }

  if (_params->isBValid()) {
    ty.vec() += _params->b().val.vec();
  }

  val.vec() = ty.vec().unaryExpr(_activate);
}

template<int PARAM_COUNT>
void MultiNode<PARAM_COUNT>::forward(Graph *graph,
    const std::array<PNode, PARAM_COUNT> &node_ptrs) {
  for (int i = 0; i < PARAM_COUNT; ++i) {
    _input_node_ptrs[i] = node_ptrs.at(i);
    node_ptrs.at(i)->parents.push_back(this);
  }
  loss = 0;
  degree = PARAM_COUNT;
  graph->addNode(this);
}

template<int PARAM_COUNT>
void MultiNode<PARAM_COUNT>::backward() {
	assert(_derivate != NULL);
  lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), _derivate);

  for (int i = 0; i < PARAM_COUNT; ++i) {
	  //cout << "MulNode<" << PARAM_COUNT<< ">backward i:"<<i <<" tag:"<< tag<<"input vec:" <<_input_node_ptrs[i]->val.vec() << endl;
    _params->w(i).grad.mat() += lty.mat() * _input_node_ptrs[i]->val.tmat();
  }

  if (_params->isBValid()) {
    _params->b().grad.vec() += lty.vec();
  }

  for (int i = 0; i < PARAM_COUNT; ++i) {
    _input_node_ptrs[i]->loss.mat() += _params->w(i).val.mat().transpose() * lty.mat();
  }
}

template<int PARAM_COUNT>
class MultiExecute : public Execute {
public:
  void forward() override;
  void backward() override;
};

template<int PARAM_COUNT>
void MultiExecute<PARAM_COUNT>::forward() {
  for (Node *ptr : batch) {
   static_cast<MultiNode<PARAM_COUNT> *>(ptr)->compute();
  }
}

template<int PARAM_COUNT>
void MultiExecute<PARAM_COUNT>::backward() {
  for (Node *ptr : batch) {
   static_cast<MultiNode<PARAM_COUNT> *>(ptr)->backward();
  }
}

template<int PARAM_COUNT>
PExecute MultiNode<PARAM_COUNT>::generate() {
  MultiExecute<PARAM_COUNT> *exec = new MultiExecute<PARAM_COUNT>();
  exec->batch.push_back(this);
  return exec;
}

#endif
