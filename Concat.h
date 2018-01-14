#ifndef CONCAT
#define CONCAT

/*
*  Concat.h:
*  concatenatation operation.
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/


#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#if USE_GPU
#include "n3ldg_cuda.h"
#endif

class ConcatNode : public Node {
public:
    vector<int> inDims;
    vector<PNode> ins;
#if USE_GPU
    n3ldg_cuda::IntArray dInOffsets;
    n3ldg_cuda::NumberPointerArray dInValues;
#endif

    ConcatNode() : Node() {
        inDims.clear();
        ins.clear();
        node_type = "concat";
    }

#if USE_GPU
    void initDeviceMembers() {
        std::vector<int> inOffsets;
        inOffsets.resize(ins.size());
        inOffsets.at(0) = 0;
        for (int i = 1; i < ins.size(); ++i) {
            inOffsets.at(i) = inOffsets.at(i - 1) + ins.at(i - 1)->dim;
        }

        dInOffsets.init(inOffsets.data(), inDims.size());
        std::vector<dtype*> vals;
        for (PNode p : ins) {
            vals.push_back(p->val.value);
        }
        dInValues.init(vals.data(), ins.size());
    }
#endif

    void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for concat" << std::endl;
            return;
        }

        ins.clear();
        for (int i = 0; i < x.size(); i++) {
            ins.push_back(x[i]);
        }

        degree = 0;
        int nSize = ins.size();
        for (int i = 0; i < nSize; ++i) {
            ins[i]->addParent(this);
        }

#if USE_GPU
        initDeviceMembers();
#endif
        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1) {
        ins.clear();
        ins.push_back(x1);

        degree = 0;
        for (int i = 0; i < 1; ++i) {
            ins[i]->addParent(this);
        }
#if USE_GPU
        initDeviceMembers();
#endif

        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2) {
        ins.clear();
        ins.push_back(x1);
        ins.push_back(x2);

        degree = 0;
        for (int i = 0; i < 2; ++i) {
            ins[i]->addParent(this);
        }

#if USE_GPU
        initDeviceMembers();
#endif
        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
        ins.clear();
        ins.push_back(x1);
        ins.push_back(x2);
        ins.push_back(x3);

        degree = 0;
        for (int i = 0; i < 3; ++i) {
            ins[i]->addParent(this);
        }

#if USE_GPU
        initDeviceMembers();
#endif
        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
        ins.clear();
        ins.push_back(x1);
        ins.push_back(x2);
        ins.push_back(x3);
        ins.push_back(x4);

        degree = 0;
        for (int i = 0; i < 4; ++i) {
            ins[i]->addParent(this);
        }

#if USE_GPU
        initDeviceMembers();
#endif
        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5) {
        ins.clear();
        ins.push_back(x1);
        ins.push_back(x2);
        ins.push_back(x3);
        ins.push_back(x4);
        ins.push_back(x5);

        degree = 0;
        for (int i = 0; i < 5; ++i) {
            ins[i]->addParent(this);
        }

#if USE_GPU
        initDeviceMembers();
#endif
        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6) {
        ins.clear();
        ins.push_back(x1);
        ins.push_back(x2);
        ins.push_back(x3);
        ins.push_back(x4);
        ins.push_back(x5);
        ins.push_back(x6);

        degree = 0;
        for (int i = 0; i < 6; ++i) {
            ins[i]->addParent(this);
        }

#if USE_GPU
        initDeviceMembers();
#endif
        cg->addNode(this);
    }

    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        if (!Node::typeEqual(other)) {
            return false;
        }
        ConcatNode *o = static_cast<ConcatNode*>(other);
        if (!isEqual(drop_value, o->drop_value)) {
            return false;
        }
        if (inDims.size() != o->inDims.size()) {
            return false;
        }
        for (int i = 0; i < inDims.size(); ++i) {
            if (inDims.at(i) != o->inDims.at(i)) {
                return false;
            }
        }
        return true;
    }

    void compute() {
        int nSize = ins.size();
        inDims.clear();
        int curDim = 0;
        for (int i = 0; i < nSize; ++i) {
            inDims.push_back(ins[i]->val.dim);
            curDim += inDims[i];
        }
        if (curDim != dim) {
            std::cout << "input dim size not match" << curDim << "\t" << dim << std::endl;
            return;
        }

        int offset = 0;
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < inDims[i]; idx++) {
                val[offset + idx] = ins[i]->val[idx];
            }
            offset += inDims[i];
        }
    }


    void backward() {
        int nSize = ins.size();
        int offset = 0;
        for (int i = 0; i < nSize; ++i) {
            for (int idx = 0; idx < inDims[i]; idx++) {
                ins[i]->loss[idx] += loss[offset + idx];
            }
            offset += inDims[i];
        }
    }

};

#if USE_GPU
class ConcatExecute : public Execute {
  public:
    bool bTrain;
    int outDim;
    int *inOffsets;
    int inCount;
    Tensor2D drop_mask;
  public:
    inline void  forward() {
        int count = batch.size();
        drop_mask.initOnDevice(outDim, count);
        n3ldg_cuda::CalculateDropoutMask(drop_factor, count, outDim,
                drop_mask.value);
        std::vector<dtype**> ins;
        ins.reserve(count);
        for (int i = 0; i < count; ++i) {
            ConcatNode *n = static_cast<ConcatNode*>(batch[i]);
#if TEST_CUDA
            for (int j = 0; j < inCount; ++j) {
                n->ins[j]->val.copyFromHostToDevice();
            }
#endif
            ins.push_back(n->dInValues.value);
        }
        std::vector<dtype*> outs;
        outs.resize(count);
        for (int i = 0; i < count; ++i) {
            ConcatNode *n = static_cast<ConcatNode*>(batch[i]);
            outs.push_back(n->val.value);
        }
        n3ldg_cuda::ConcatForward(ins, inOffsets, outs, count, inCount,
                outDim);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
            assert(batch[idx]->val.verify("concat forward"));
        }
#endif
    }

    inline void backward() {
        int count = batch.size();
    }
};
#else
class ConcatExecute : public Execute {
  public:
    bool bTrain;
  public:
    inline void  forward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            batch[idx]->forward_drop(bTrain, drop_factor);
        }
    }

    inline void backward() {
        int count = batch.size();
        //#pragma omp parallel for
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward_drop();
            batch[idx]->backward();
        }
    }
};
#endif

inline PExecute ConcatNode::generate(bool bTrain, dtype cur_drop_factor) {
    ConcatExecute* exec = new ConcatExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor * drop_value;
#if USE_GPU
    exec->inCount = this->ins.size();
    exec->inOffsets = this->dInOffsets.value;
#endif
    exec->outDim = 0;
    for (int d : inDims) {
        exec->outDim += d;
    }
    return exec;
}

#endif
