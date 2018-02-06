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
#include "profiler.h"

class ConcatNode : public Node {
public:
    vector<int> inDims;
    vector<PNode> ins;

    ConcatNode() : Node() {
        inDims.clear();
        ins.clear();
        node_type = "concat";
    }

#if USE_GPU
    void toNodeInfo(NodeInfo &info) const override {
        Node::toNodeInfo(info);
        for (PNode p : ins) {
            info.input_vals.push_back(p->val.value);
            info.input_losses.push_back(p->loss.value);
            info.input_dims.push_back(p->dim);
        }
    }
#endif

    void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for concat" << std::endl;
            abort();
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
        inDims.clear();
        int curDim = 0;
        for (int i = 0; i < nSize; ++i) {
            inDims.push_back(ins[i]->val.dim);
            curDim += inDims[i];
        }
        if (curDim != dim) {
            std::cout << "input dim size not match" << curDim << "\t" << dim << std::endl;
            abort();
        }
        cg->addNode(this);
    }

    void forward(Graph *cg, PNode x1) {
        ins.clear();
        ins.push_back(x1);

        degree = 0;
        for (int i = 0; i < 1; ++i) {
            ins[i]->addParent(this);
        }
        inDims.clear();
        int curDim = 0;
        for (int i = 0; i < ins.size(); ++i) {
            inDims.push_back(ins[i]->val.dim);
            curDim += inDims[i];
        }
        if (curDim != dim) {
            std::cout << "input dim size not match" << curDim << "\t" << dim << std::endl;
            abort();
        }

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

        inDims.clear();
        int curDim = 0;
        for (int i = 0; i < ins.size(); ++i) {
            inDims.push_back(ins[i]->val.dim);
            curDim += inDims[i];
        }
        if (curDim != dim) {
            std::cout << "input dim size not match" << curDim << "\t" << dim << std::endl;
            abort();
        }
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
        inDims.clear();
        int curDim = 0;
        for (int i = 0; i < ins.size(); ++i) {
            inDims.push_back(ins[i]->val.dim);
            curDim += inDims[i];
        }
        if (curDim != dim) {
            std::cout << "input dim size not match" << curDim << "\t" << dim << std::endl;
            abort();
        }

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
        inDims.clear();
        int curDim = 0;
        for (int i = 0; i < ins.size(); ++i) {
            inDims.push_back(ins[i]->val.dim);
            curDim += inDims[i];
        }
        if (curDim != dim) {
            std::cout << "input dim size not match" << curDim << "\t" << dim << std::endl;
            abort();
        }

        cg->addNode(this);
    }

    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) {
        if (!Node::typeEqual(other)) {
            return false;
        }
        ConcatNode *o = static_cast<ConcatNode*>(other);
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

    size_t typeHashCode() const override {
        size_t hash_code = Node::typeHashCode() ^
            std::hash<int>{}(inDims.size());
        int i = 0;
        for (int dim : inDims) {
            hash_code ^= (dim << (i++ % 16));
        }
        return hash_code;
    }

    void compute() {
        int nSize = ins.size();
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
    int inCount;
    Tensor2D drop_mask;
  public:
    void  forward() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent("ConcatNode forward");
        int count = batch.size();
        assert(drop_factor < 1);
        if (drop_factor > 0) {
#if TEST_CUDA
            drop_mask.init(outDim, count);
#else
            drop_mask.initOnDevice(outDim, count);
#endif
            n3ldg_cuda::CalculateDropoutMask(drop_factor, count, outDim,
                    drop_mask.value);
        }
        n3ldg_cuda::ConcatForward(graph_info, drop_mask.value, drop_factor,
            count, inCount, outDim);
#if TEST_CUDA
        if (drop_factor > 0) {
            drop_mask.copyFromDeviceToHost();
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < outDim; ++j) {
                    dtype v = drop_mask[j][i];
                    batch[i]->drop_mask[j] = v <= drop_factor ? 0 : 1;
                }
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            if (drop_factor > 0) {
                batch[idx]->forward_drop(bTrain, drop_factor /
                        batch[0]->drop_value);
            }
            n3ldg_cuda::Assert(batch[idx]->val.verify("concat forward"));
        }
#endif
        profiler.EndCudaEvent();
    }

    void backward() {
        int count = batch.size();
        std::vector<dtype**> in_losses;
        in_losses.reserve(count);
        for (int i = 0; i < count; ++i) {
            ConcatNode *n = static_cast<ConcatNode*>(batch[i]);
#if TEST_CUDA
            for (int j = 0; j < inCount; ++j) {
                //n->ins[j]->loss.copyFromHostToDevice();
            }
#endif
            //in_losses.push_back(n->dInLosses.value); // TODO
        }
        std::vector<dtype*> out_losses;
        out_losses.reserve(count);
        for (int i = 0; i < count; ++i) {
            ConcatNode *n = static_cast<ConcatNode*>(batch[i]);
#if TEST_CUDA
            //n->loss.copyFromHostToDevice();
            for (int j=0; j < inCount; ++j) {
                //n->ins[j]->loss.copyFromHostToDevice();
            }
#endif
            out_losses.push_back(n->loss.value);
        }

        n3ldg_cuda::ConcatBackward(this->graph_info, drop_mask.value,
                drop_factor, count, inCount, outDim);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            if (drop_factor > 0) {
                batch[idx]->backward_drop();
            }
            batch[idx]->backward();
        }
        for (int idx = 0; idx < count; idx++) {
            for (int j = 0; j < inCount; ++j) {
                n3ldg_cuda::Assert(static_cast<ConcatNode *>(batch[idx])->
                        ins[j]->loss.verify("concat backward"));
            }
        }
#endif
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
    exec->outDim = 0;
    for (int d : inDims) {
        exec->outDim += d;
    }
#endif
    return exec;
}

#endif
