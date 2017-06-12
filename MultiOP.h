#ifndef N3LDG_MULTI_PARAMS_H
#define N3LDG_MULTI_PARAMS_H

#include <array>
#include <boost/range/irange.hpp>
#include "Param.h"
#include "ModelUpdate.h"
#include "MyLib.h"

template<int PARAM_COUNT>
class MultiParams {
public:
  MultiParams() : _isBValid(true) {}
  virtual ~MultiParams() = default;

  void exportToAdaParams(ModelUpdate &ada);
  void save(std::ofstream &os) const;
  void load(std::ifstream &is, AlignedMemoryPool *pool = NULL);

protected:
  void initial(int oSize, const std::array<int, PARAM_COUNT> &inSizeArray,
      bool isBValid = true,
      AlignedMemoryPool* pool = NULL);
  std::array<Param, PARAM_COUNT> _params;
  Param _b;
  bool _isBValid;
};

template<int PARAM_COUNT>
void MultiParams<PARAM_COUNT>::initial(int oSize,
    const std::array<int, PARAM_COUNT> &inSizeArray,
    bool isBValid,
    AlignedMemoryPool* pool) {
  for (int i : boost::irange(0, PARAM_COUNT)) {
    _params.at(i).initial(oSize, inSizeArray[i], pool);
  }
  if (isBValid) {
    _b.initial(oSize, 1, pool);
  }
}

template<int PARAM_COUNT>
void MultiParams<PARAM_COUNT>::exportToAdaParams(ModelUpdate &ada) {
  for (Param *param : _params) {
    ada.addParam(param);
  }
  if (_isBValid) {
    ada.addParam(_b);
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


#endif
