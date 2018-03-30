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
        std::vector<PNode> ins = {x1};
        forward(cg, ins);
    }

    void forward(Graph *cg, PNode x1, PNode x2) {
        std::vector<PNode> ins = {x1, x2};
        forward(cg, ins);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3) {
        std::vector<PNode> ins = {x1, x2, x3};
        forward(cg, ins);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4) {
        std::vector<PNode> ins = {x1, x2, x3, x4};
        forward(cg, ins);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5) {
        std::vector<PNode> ins = {x1, x2, x3, x4, x5};
        forward(cg, ins);
    }

    void forward(Graph *cg, PNode x1, PNode x2, PNode x3, PNode x4, PNode x5, PNode x6) {
        std::vector<PNode> ins = {x1, x2, x3, x4, x5, x6};
        forward(cg, ins);
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
            for (int idx = 0; idx < inDims.at(i); idx++) {
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
    int outDim;
    int inCount;
    Tensor2D drop_mask;

    void  forward() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent("ConcatNode forward");
        int count = batch.size();
        drop_mask.init(outDim, count);
        CalculateDropMask(count, outDim, drop_mask);
        n3ldg_cuda::ConcatForward(graph_info, bTrain, drop_mask.value,
            dynamicDropValue(), count, inCount, outDim);
#if TEST_CUDA
        if (initialDropValue() > 0) {
            drop_mask.copyFromDeviceToHost();
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < outDim; ++j) {
                    dtype v = drop_mask[j][i];
                    batch[i]->drop_mask[j] = v <= dynamicDropValue() ? 0 : 1;
                }
            }
        }
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            if (initialDropValue() > 0) {
                batch[idx]->forward_drop(bTrain, drop_factor);
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
        std::vector<dtype*> out_losses;
        out_losses.reserve(count);
        for (int i = 0; i < count; ++i) {
            ConcatNode *n = static_cast<ConcatNode*>(batch[i]);
            out_losses.push_back(n->loss.value);
        }

        n3ldg_cuda::ConcatBackward(this->graph_info, drop_mask.value,
                dynamicDropValue(), count, inCount, outDim);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            if (initialDropValue() > 0) {
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
    exec->drop_factor = cur_drop_factor;
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
