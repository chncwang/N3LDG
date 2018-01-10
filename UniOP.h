#ifndef UNIOP_H_
#define UNIOP_H_

/*
*  UniOP.h:
*  a simple feed forward neural operation, unary input.
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/


#include "Param.h"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"
#include "profiler.h"

class UniParams {
  public:
    Param W;
    Param b;
    bool bUseB;

  public:
    UniParams() {
        bUseB = true;
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

    inline void initial(int nOSize, int nISize, bool useB = true) {
        W.initial(nOSize, nISize);

        bUseB = useB;
        if (bUseB) {
            b.initial(nOSize, 1);
        }
    }

    inline void save(std::ofstream &os) const {
        os << bUseB << std::endl;
        W.save(os);
        if (bUseB) {
            b.save(os);
        }
    }

    inline void load(std::ifstream &is) {
        is >> bUseB;
        W.load(is);
        if (bUseB) {
            b.load(is);
        }
    }

};


// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class UniNode : public Node {
  public:
    PNode in;
    UniParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    Tensor1D ty, lty;


  public:
    UniNode() : Node() {
        in = NULL;
        activate = ftanh;
        derivate = dtanh;
        param = NULL;
        node_type = "uni";
    }

    ~UniNode() {
        in = NULL;
    }

    inline void init(int ndim, dtype dropout) {
        Node::init(ndim, dropout);
        ty.init(ndim);
        lty.init(ndim);
    }


    inline void setParam(UniParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
        ty = 0;
        lty = 0;
    }

    // define the activate function and its derivation form
    inline void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
        activate = f;
        derivate = f_deri;
    }

  public:
    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        ty.mat() = param->W.val.mat() * in->val.mat();
        if (param->bUseB) {
            ty.vec() += param->b.val.vec();
        }
        val.vec() = ty.vec().unaryExpr(ptr_fun(activate));
    }

    inline void backward() {
        lty.vec() = loss.vec() * ty.vec().binaryExpr(val.vec(), ptr_fun(derivate));
        param->W.grad.mat() += lty.mat() * in->val.tmat();
        if (param->bUseB) {
            param->b.grad.vec() += lty.vec();
        }
        in->loss.mat() += param->W.val.mat().transpose() * lty.mat();
    }

  public:
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        UniNode* conv_other = (UniNode*)other;
        if (param != conv_other->param) {
            return false;
        }
        if (activate != conv_other->activate || derivate != conv_other->derivate) {
            return false;
        }
        if (!isEqual(drop_value, other->drop_value)) {
            return false;
        }

        return true;
    }

};

// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class LinearUniNode : public Node {
  public:
    PNode in;
    UniParams* param;

  public:
    LinearUniNode() : Node() {
        in = NULL;
        param = NULL;
        node_type = "linear_uni";
    }


    inline void setParam(UniParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
    }


  public:
    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        val.mat() = param->W.val.mat() * in->val.mat();
        if (param->bUseB) {
            val.vec() += param->b.val.vec();
        }
    }

    inline void backward() {
        param->W.grad.mat() += loss.mat() * in->val.tmat();
        if (param->bUseB) {
            param->b.grad.vec() += loss.vec();
        }
        in->loss.mat() += param->W.val.mat().transpose() * loss.mat();
    }

  public:
    inline PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        LinearUniNode* conv_other = (LinearUniNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};



// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
class LinearNode : public Node {
  public:
    PNode in;
    UniParams* param;

  public:
    LinearNode() : Node() {
        in = NULL;
        param = NULL;
        node_type = "linear";
    }


    inline void setParam(UniParams* paramInit) {
        param = paramInit;
    }

    inline void clearValue() {
        Node::clearValue();
        in = NULL;
    }


  public:
    void forward(Graph *cg, PNode x) {
        in = x;
        degree = 0;
        in->addParent(this);
        cg->addNode(this);
    }

  public:
    inline void compute() {
        val.mat() = param->W.val.mat() * in->val.mat();
    }

    inline void backward() {
        param->W.grad.mat() += loss.mat() * in->val.tmat();
        in->loss.mat() += param->W.val.mat().transpose() * loss.mat();
    }

  public:
    PExecute generate(bool bTrain, dtype cur_drop_factor);

    // better to rewrite for deep understanding
    inline bool typeEqual(PNode other) {
        bool result = Node::typeEqual(other);
        if (!result) return false;
        LinearNode* conv_other = (LinearNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

};


class UniExecute :public Execute {
  public:
    Tensor2D x, ty, y;
    int inDim, outDim;
    UniParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    Tensor2D drop_mask;
    bool bTrain;

    inline void  forward() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent("forward");
        int count = batch.size();

//        profiler.BeginEvent("init");
        ty.init(outDim, count);
        x.init(inDim, count);
        y.init(outDim, count);
        drop_mask.init(outDim, count);
#if USE_GPU
//        profiler.EndCudaEvent();
#else
        Tensor2D b;
        b.init(outDim, count);
//        profiler.EndEvent();
#endif

#if USE_GPU
        std::vector<dtype*> xs, ys;
        xs.reserve(batch.size());
        ys.reserve(batch.size());

//        profiler.BeginEvent("copy between device and host");
//        param->W.val.copyFromHostToDevice();
//        param->b.val.copyFromHostToDevice();
//        profiler.EndCudaEvent();

        for (int i = 0; i < batch.size(); ++i) {
            UniNode *n = static_cast<UniNode*>(batch.at(i));

//            profiler.BeginEvent("copy between device and host");
            //n->in->val.copyFromHostToDevice();
//            profiler.EndCudaEvent();

//            profiler.BeginEvent("vector push_back");
            xs.push_back(n->in->val.value);
            ys.push_back(n->val.value);
//            profiler.EndEvent();
        }

//        profiler.BeginEvent("CopyForUniNodeForward");
        n3ldg_cuda::CopyForUniNodeForward(xs, param->b.val.value, x.value,
                ty.value,
                count,
                inDim,
                outDim);
//        profiler.EndCudaEvent();

//        profiler.BeginEvent("MatrixMultiplyMatrix");
        n3ldg_cuda::MatrixMultiplyMatrix(param->W.val.value, x.value,
                ty.value,
                outDim,
                inDim,
                count,
                param->bUseB);
//        profiler.EndCudaEvent();

        profiler.BeginEvent("dropout mask");
        n3ldg_cuda::CalculateDropoutMask(drop_factor, count, outDim,
                drop_mask.value);
        profiler.EndCudaEvent();

//        profiler.BeginEvent("Tanh");
        n3ldg_cuda::Tanh(ty.value, ys, y.value, outDim);
//        profiler.EndCudaEvent();

        for (int i = 0; i<batch.size(); ++i) {
            UniNode *n = static_cast<UniNode*>(batch.at(i));

//            profiler.BeginEvent("copy between device and host");
//            n->val.copyFromDeviceToHost();
//            profiler.EndCudaEvent();
        }

//        profiler.BeginEvent("copy between device and host");
        //x.copyFromDeviceToHost();
        //y.copyFromDeviceToHost();
        //ty.copyFromDeviceToHost();
//        profiler.EndCudaEvent();

        profiler.EndCudaEvent();
#else
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                x[idy][idx] = ptr->in->val[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idy][idx] = param->b.val.v[idy];
                }
            }
        }

        ty.mat() = param->W.val.mat() * x.mat();

        if (param->bUseB) {
            ty.vec() = ty.vec() + b.vec();
        }

        y.vec() = ty.vec().unaryExpr(ptr_fun(activate));

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idy][idx];
            }
        }

//        x.verify();
//        ty.verify();
//        y.verify();
        profiler.EndEvent();
#endif
    }

    void backward() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
//        profiler.BeginEvent("backward");
        int count = batch.size();
        Tensor2D lx, lty, ly;
        lx.init(inDim, count);
        lty.init(outDim, count);
        ly.init(outDim, count);

#if USE_GPU
        std::vector<dtype*> ly_vec;
        ly_vec.reserve(count);
        for (int i = 0; i < count; ++i) {
            UniNode* ptr = (UniNode*)batch[i];
            //ptr->loss.copyFromHostToDevice();
            ly_vec.push_back(ptr->loss.value);
        }

//        profiler.BeginEvent("cal lty");
        n3ldg_cuda::CalculateLtyForUniBackward(ly_vec, ty.value,
                y.value, lty.value, count, outDim);
//        profiler.EndCudaEvent();
//        profiler.BeginEvent("copy");
        //param->W.grad.copyFromHostToDevice();
//        profiler.EndCudaEvent();
//        profiler.BeginEvent("cal W grad");
        n3ldg_cuda::MatrixMultiplyMatrix(lty.value, x.value,
                param->W.grad.value, outDim, count, inDim, true, true, false);
//        profiler.EndCudaEvent();
//        profiler.BeginEvent("copy");
        //param->W.val.copyFromHostToDevice();
//        profiler.EndCudaEvent();
//        profiler.BeginEvent("cal lx");
        n3ldg_cuda::MatrixMultiplyMatrix(param->W.val.value, lty.value,
                lx.value, inDim, outDim, count, false, false, true);
//        profiler.EndCudaEvent();
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
//            ptr->in->loss.copyFromHostToDevice();
            losses.push_back(ptr->in->loss.value);
        }

//        profiler.BeginEvent("copy");
        //param->b.grad.copyFromHostToDevice();
//        profiler.EndCudaEvent();
//        profiler.BeginEvent("add bias and losses");
        n3ldg_cuda::AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(
                lty.value, lx.value, param->b.grad.value, losses, count,
                outDim, inDim);
//        profiler.EndCudaEvent();
//        profiler.EndCudaEvent();
#else
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            ptr->backward_drop();
            for (int idy = 0; idy < outDim; idy++) {
                ly[idy][idx] = ptr->loss[idy];
            }
        }

        lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));
        //lty.verify();
        //x.verify();

        param->W.grad.mat() += lty.mat() * x.mat().transpose();
        //param->W.grad.verify();

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += lty[idy][idx];
                }
            }
        }
        //param->b.grad.verify();

        lx.mat() += param->W.val.mat().transpose() * lty.mat();
        //lx.verify();

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss[idy] += lx[idy][idx];
            }
        }

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val[idy] = y[idx][idy];
            }
            ptr->forward_drop(bTrain, drop_factor);
        }

//        for (Node * n : batch) {
//            UniNode *ptr = static_cast<UniNode *>(n);
//            ptr->in->loss.verify();
//        }
//        profiler.EndEvent();
#endif
    }
};

inline PExecute UniNode::generate(bool bTrain, dtype cur_drop_factor) {
    UniExecute* exec = new UniExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    exec->inDim = param->W.inDim();
    exec->outDim = param->W.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;
    exec->drop_factor = cur_drop_factor * drop_value;
    return exec;
};

class LinearUniExecute :public Execute {
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

inline PExecute LinearUniNode::generate(bool bTrain, dtype cur_drop_factor) {
    LinearUniExecute* exec = new LinearUniExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};

class LinearExecute :public Execute {
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

inline PExecute LinearNode::generate(bool bTrain, dtype cur_drop_factor) {
    LinearExecute* exec = new LinearExecute();
    exec->batch.push_back(this);
    exec->bTrain = bTrain;
    exec->drop_factor = cur_drop_factor;
    return exec;
};


#endif /* UNIOP_H_ */
