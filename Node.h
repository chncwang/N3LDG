#ifndef BasicNode
#define BasicNode

/*
*  Node.h:
*  basic processing unit in a neural network
*  (1) we have a node structure to build user graph
*  (2) we have a execute structure to merge similar nodes that can be execute together
*  The real forward and backward are defined in Execute.
*  Every operation should define a node class and a execute class together.
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/

#include "MyTensor.h"

#include <array>
#include <string>

enum NodeType {
    SCALAR_MULTIPLY = 0,
    ACTIVATED = 1
};

const std::string &toString(NodeType node_type) {
    static std::array<std::string, 2> node_type_strings = { "scalar_multiply", "activated" };
    return node_type_strings.at(node_type);
}

class Execute;

// one Node means a vector
// the col should be 1, because we aimed for NLP only
class Node {
public:
    vector<Node*> parents;
public:
    Tensor1D val;
    Tensor1D loss;
public:
    int dim;
    int degree;
    string node_type;

public:
    Tensor1D drop_mask;
    dtype drop_value;


public:
    Node() {
        dim = 0;
        degree = 0;
        parents.clear();
        node_type = "interface";
        drop_value = -1;
    }

    virtual ~Node() {
        val = 0;
        loss = 0;
        degree = 0;
        parents.clear();
        node_type.clear();
    }


public:
    virtual inline void clearValue() {
        val = 0;
        loss = 0;
        degree = 0;
        if (drop_value > 0)drop_mask = 1;
        parents.clear();
    }

    virtual inline void init(int ndim, dtype dropout) {
        n3ldg_assert(ndim > 0, "dim should be a positive integer but is " << ndim);
        dim = ndim;
        val.init(dim);
        loss.init(dim);
        drop_mask.init(dim);
        if (dropout > 0 && dropout <= 1) {
            drop_value = dropout;
        } else {
            drop_value = -1;
        }
        parents.clear();
    }

    inline void forward_drop(bool bTrain) {
        if (drop_value > 0) {
            if (bTrain) {
                int dropNum = (int)(dim * drop_value);
                vector<int> tmp_masks(dim);
                for (int idx = 0; idx < dim; idx++) {
                    tmp_masks[idx] = idx < dropNum ? 0 : 1;
                }
                random_shuffle(tmp_masks.begin(), tmp_masks.end());
                for (int idx = 0; idx < dim; idx++) {
                    drop_mask[idx] = tmp_masks[idx];
                }
                val.vec() = val.vec() * drop_mask.vec();
            } else {
                val.vec() = val.vec() * (1 - drop_value);
            }
        }
        degree = -1;
    }

    inline void backward_drop() {
        if (drop_value > 0) {
            loss.vec() = loss.vec() * drop_mask.vec();
        }
    }

public:
    virtual void compute() {}
    virtual void backward() {}

    virtual inline Execute* generate(bool bTrain) = 0;

    virtual inline bool typeEqual(Node* other) {
        if (node_type.compare(other->node_type) == 0) {
            return true;
        }
        return false;
    }

public:
    virtual inline void addParent(Node* parent) {
        if (degree >= 0) {
            parents.push_back(parent);
            parent->degree++;
        }
    }


};

typedef  Node* PNode;


class Execute {
public:
    vector<PNode> batch;

public:
    virtual ~Execute() {
        batch.clear();
    }

public:
    virtual void forward() = 0;
    virtual void backward() = 0;


    virtual inline bool addNode(PNode in) {
        if (batch.empty()) {
            std::cout << "empty batch, strange...." << std::endl;
            return false;
        }

        if (batch[0]->typeEqual(in)) {
            batch.push_back(in);
            return true;
        }

        return false;
    }
};

class DropoutableExecute : public Execute {
public:
    void init(PNode pnode, bool btrain) {
        batch.push_back(pnode);
        bTrain = btrain;
    }

    void forward() override {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            Node* ptr = batch[idx];
            ptr->compute();
            ptr->forward_drop(bTrain);
        }
    }

    void backward() override {
        int count = batch.size();
        for (int idx = 0; idx < count; idx++) {
            Node* ptr = batch[idx];
            ptr->backward_drop();
            ptr->backward();
        }
    }

private:
    bool bTrain;
};

class DropoutableNode :public Node {
public:
    Execute* generate(bool bTrain) override {
        n3ldg_assert(drop_value > 0 && drop_value < 1, "dropout is invalid");
        DropoutableExecute * execute = new DropoutableExecute();
        execute->init(this, bTrain);
    }
};

typedef  Execute* PExecute;

#endif
