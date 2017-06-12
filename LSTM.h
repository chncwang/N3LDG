#ifndef N3LDG_LSTM_H
#define N3LDG_LSTM_H

#include "Param.h"
#include "MyLIb.h"
#include "Node.h"
#include "TriOP.h"
#include "BiOP.h"
#include "Graph.h"

class LSTMParams {
public:
  LSTMParams() = default;

private:
  TriParams _inputParams;
  TriParams _outputParams;
  TriParams _forgetParams;
  BiParams _cellParams;
};

#endif
