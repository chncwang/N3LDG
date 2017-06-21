#ifndef N3LDG_LSTM_H
#define N3LDG_LSTM_H

#include "Param.h"
#include "MyLIb.h"
#include "Node.h"
#include "TriOP.h"
#include "BiOP.h"
#include "Graph.h"
#include "AddOP.h"
#include "TanhOP.h"

struct LSTMParams {
  LSTMParams() = default;
  ~LSTMParams() = default;

  void initial(int outDim, int inDim, AlignedMemoryPool *pool = NULL);
  void exportToAdaParams(ModelUpdate &ada);
  int inDim();
  int outDim();

  TriParams inputParams;
  TriParams outputParams;
  TriParams forgetParams;
  BiParams cellParams;
};

void LSTMParams::initial(int outDim, int inDim, AlignedMemoryPool *pool) {
	inputParams.initial(outDim, {inDim, outDim, outDim}, true, pool);
	forgetParams.initial(outDim, {inDim, outDim, outDim});
	cellParams.initial(outDim, {inDim, outDim}, true, pool);
	outputParams.initial(outDim, { inDim, outDim, outDim }, true, pool);
}

void LSTMParams::exportToAdaParams(ModelUpdate &ada) {
  inputParams.exportToAdaParams(ada);
  outputParams.exportToAdaParams(ada);
  forgetParams.exportToAdaParams(ada);
  cellParams.exportToAdaParams(ada);
}

int LSTMParams::inDim() {
  return inputParams.w1().inDim();
}

int LSTMParams::outDim() {
  return inputParams.w3().inDim();
}

/* *
 * Standard convolutional LSTM builder.
 * see http://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf
 * */
class LSTMBuilder {
  public:
    LSTMBuilder();
    ~LSTMBuilder() = default;

    void clear();
    void init(LSTMParams *params, dtype dropoutRatio, bool shouldLeftToRight = true, AlignedMemoryPool *pool = NULL);
    void resize(int maxSize);
    void forward(Graph *cg, const vector<PNode>& x, int words_num);

    void forwardFromLeftToRight(Graph *graph, const vector<PNode> &x, int words_num);
    void forwardFromRightToLeft(Graph *graph, const vector<PNode> &x);

    int _nSize;

    vector<TriNode> _inputGates;
   vector<TriNode> _forgetGates;
    vector<BiNode> _halfCells;

   vector<PMultiNode> _inputFilters;
    vector<PMultiNode> _forgetFilters;
    vector<PAddNode> _cells;
    vector<TriNode> _outputGates;
    vector<TanhNode> _halfHiddens;
    vector<PMultiNode> _hiddens;
    BucketNode _bucketHiddenNode;
	BucketNode _bucketCellNode;
    LSTMParams *_params;

    bool _shouldLeftToRight;
};

LSTMBuilder::LSTMBuilder() {
  clear();
}

void LSTMBuilder::clear() {
	for (TriNode &n : _inputGates) {
		n.clearValue();
	}
	_inputGates.clear();

	for (TriNode &n : _forgetGates) {
		n.clearValue();
	}
	_forgetGates.clear();

	for (BiNode &n : _halfCells){
		n.clearValue();
	}
	_halfCells.clear();

	for (PMultiNode &n : _inputFilters) {
		_inputFilters.clear();
	}

	for (PMultiNode &n : _forgetFilters) {
		_forgetFilters.clear();
	}

	for (PAddNode &n : _cells) {
	_cells.clear();
}
	for (TriNode &n : _outputGates) {
		n.clearValue();
	}
  _outputGates.clear();

  for (TanhNode &n : _halfHiddens) {
	  _halfHiddens.clear();
  }
  _hiddens.clear();

  _shouldLeftToRight = true;
  _params = NULL;
  _nSize = 0;
}

void LSTMBuilder::init(LSTMParams *params, dtype dropoutRatio, bool shouldLeftToRight, AlignedMemoryPool *pool) {
  _params = params;
  int upperLimitSize = _inputFilters.size();

  for (int i = 0; i < upperLimitSize; ++i) {

	  _inputGates.at(i).tag = "lstm input " + std::to_string(i);
	  _inputFilters.at(i).tag = "lstm input filter " + std::to_string(i);
	  _forgetGates.at(i).tag = "lstm forget " + std::to_string(i);
	  _outputGates.at(i).tag = "lstm output " + std::to_string(i);
	  _halfCells.at(i).tag = "lstm half cells " + std::to_string(i);
	  _cells.at(i).tag = "lstm cells " + std::to_string(i);
	  _bucketCellNode.tag = "lstm bucket cell";
	  _bucketHiddenNode.tag = "lstm bucket hidden";
	  _hiddens.at(i).tag = "lstm hidden " + std::to_string(i);
	  _halfHiddens.at(i).tag = "lstm half hidden" + std::to_string(i);
	  _forgetFilters.at(i).tag = "lstm forget filter " + std::to_string(i);
	/*  _forgetFilters.at(i)._backword_callback_function = [i](const Vec &vec) {
		  cout << "LSTM init forget filter i:" << i << endl;
		  cout << "vec:" << vec << endl;
		  if (i == 7) {
			  cout << i << endl;
		  }
	  };
	 _cells.at(i)._backword_callback_function = [i](const Vec &vec) {
		  cout << "LSTM init cells i:" << i << endl;
		  cout << "vec:" << vec << endl;
		  if (i == 7) {
			  cout << i << endl;
		  }
		  if (i > 100) {
			  assert(false);
		  }
	  };*/
	  _inputGates.at(i).setActivationAndDerivation(fsigmoid, dsigmoid);
	  _forgetGates.at(i).setActivationAndDerivation(fsigmoid, dsigmoid);
	  _outputGates.at(i).setActivationAndDerivation(fsigmoid, dsigmoid);
	  _halfCells.at(i).setActivationAndDerivation(ftanh, dtanh);
  }

  for (int i = 0; i < upperLimitSize; ++i) {
    _inputGates.at(i).setParams(&_params->inputParams);
    _forgetGates.at(i).setParams(&_params->forgetParams);
    _outputGates.at(i).setParams(&_params->outputParams);
    _halfCells.at(i).setParams(&_params->cellParams);
  }

  _shouldLeftToRight = shouldLeftToRight;

  int outDim = _params->outDim();

  for (int i = 0; i < upperLimitSize; ++i) {
    _inputGates.at(i).init(outDim, pool);
    _forgetGates.at(i).init(outDim,  pool);
    _halfCells.at(i).init(outDim,  pool);
    _inputFilters.at(i).init(outDim,  pool);
  _forgetFilters.at(i).init(outDim,  pool);
    _cells.at(i).init(outDim, pool);
    _outputGates.at(i).init(outDim,  pool);
    _halfHiddens.at(i).init(outDim,  pool);
    _hiddens.at(i).init(outDim, pool);
  }

  _bucketHiddenNode.init(outDim, pool);
  _bucketCellNode.init(outDim, pool);
}

void LSTMBuilder::resize(int nSize) {
  _inputGates.resize(nSize);
  _forgetGates.resize(nSize);
  _halfCells.resize(nSize);
  _inputFilters.resize(nSize);
  _forgetFilters.resize(nSize);
  _cells.resize(nSize);
  _outputGates.resize(nSize);
  _halfHiddens.resize(nSize);
  _hiddens.resize(nSize);
}

void LSTMBuilder::forward(Graph *cg, const vector<PNode>& x, int words_num) {
	if (x.empty()) {
		cout << "x is empty" << endl;
		assert(false);
	}
	if (x.at(0)->val.dim != _params->inDim()) {
		cout << "dim not equal:x.at(0)->val.dim = " << x.at(0)->val.dim << " params->inDim() = " << _params->inDim() << endl;
		assert(false);
	}
  _nSize = x.size();
  if (_shouldLeftToRight) {
    forwardFromLeftToRight(cg, x, words_num);
  } else {
	  forwardFromRightToLeft(cg, x);
  }
}

void LSTMBuilder::forwardFromLeftToRight(Graph *graph,
    const vector<PNode> &x, int words_num) {
	_bucketCellNode.forward(graph, 0);
	_bucketHiddenNode.forward(graph, 0);

  _inputGates.at(0).forward(graph, {x.at(0) , &_bucketHiddenNode, &_bucketCellNode});
  _halfCells.at(0).forward(graph, {x.at(0), &_bucketHiddenNode});
  _inputFilters.at(0).forward(graph, &_inputGates.at(0), &_halfCells.at(0));
  _cells.at(0).forward(graph, {&_bucketCellNode, &_inputFilters.at(0)});
  _outputGates.at(0).forward(graph, {x.at(0), &_bucketHiddenNode, &_cells.at(0)});
  _halfHiddens.at(0).forward(graph, &_cells.at(0));
  _hiddens.at(0).forward(graph, &_halfHiddens.at(0), &_outputGates.at(0));

  for (int i=1; i< words_num; ++i) {
    _inputGates[i].forward(graph, {x.at(i), &_hiddens.at(i - 1), &_cells.at(i - 1)});
    _forgetGates[i].forward(graph, {x.at(i), &_hiddens.at(i - 1), &_cells.at(i -1)});
    _halfCells[i].forward(graph, {x[i], &_hiddens.at(i - 1)});
    _inputFilters[i].forward(graph, &_halfCells[i], &_inputGates[i]);
    _forgetFilters[i].forward(graph, &_cells[i - 1], &_forgetGates[i]);
    _cells[i].forward(graph, {&_inputFilters[i], &_forgetFilters[i]});
	_outputGates[i].forward(graph, {x.at(i), &_hiddens[i - 1], &_cells[i] });
    _halfHiddens[i].forward(graph, &_cells[i]);
    _hiddens[i].forward(graph, &_halfHiddens[i], &_outputGates[i]);
  }
}

void LSTMBuilder::forwardFromRightToLeft(Graph *graph,
    const vector<PNode> &x) {
	assert(false);
  //_inputGates.at(_nSize - 1).forward(graph, {&_bucketNode, &_bucketNode, x.at(_nSize - 1)});
  //_halfCells.at(_nSize - 1).forward(graph, {&_bucketNode, x.at(_nSize - 1)});
  //_inputFilters.at(_nSize - 1).forward(graph, &_halfCells.at(_nSize - 1), &_inputGates.at(_nSize - 1));
  //_cells.at(_nSize - 1).forward(graph, {&_inputFilters.at(_nSize - 1), &_bucketNode});
  //_halfHiddens.at(_nSize - 1).forward(graph, &_cells.at(_nSize - 1));
  //_outputGates.at(_nSize - 1).forward(graph, {&_bucketNode, &_cells.at(_nSize - 1), x.at(_nSize - 1)});
  //_hiddens.at(_nSize - 1).forward(graph, &_halfHiddens.at(_nSize - 1), &_outputGates.at(_nSize - 1));

  //for (int i=_nSize - 2; i>= 0; --i) {
  //  _inputGates[i].forward(graph, {&_hiddens[i - 1], &_cells[i - 1], x[i]});
  //  _forgetGates[i].forward(graph, {&_hiddens[i - 1], &_cells[i - 1], x[i]});
  //  _halfCells[i].forward(graph, {&_hiddens[i - 1], x[i]});
  //  _inputFilters[i].forward(graph, &_halfCells[i], &_inputGates[i]);
  //  _forgetFilters[i].forward(graph, &_cells[i - 1], &_forgetGates[i]);
  //  _cells[i].forward(graph, {&_inputFilters[i], &_forgetFilters[i]});
  //  _halfHiddens[i].forward(graph, &_cells[i]);
  //  _outputGates[i].forward(graph, {&_hiddens[i - 1], &_cells[i], x[i]});
  //  _hiddens[i].forward(graph, &_halfHiddens[i], &_outputGates[i]);
  //}
}

#endif
