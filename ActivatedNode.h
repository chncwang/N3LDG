#ifndef N3LDG_ACTIVATED_NODE
#define N3LDG_ACTIVATED_NODE

#include <functional>
#include "Node.h"

class ActivatedNode : public DropoutableNode {
public:
    PNode in = NULL;
    std::function<dtype(dtype)> activate = ftanh;
    std::function<dtype(dtype, dtype)> derivate = dtanh;

    ActivatedNode() {
        node_type = toString(NodeType::ACTIVATED);
    }

    void clearValue() override {
        in = NULL;
        Node::clearValue();
    }

    void forward(Graph *graph, PNode innode) {
        in = innode;
        degree = 0;
        in->addParent(this);
        graph->addNode(this);
    }

    void compute() override {
        val.vec() = in->val.vec().unaryExpr(activate);
    }

    void backward() override {
        in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), derivate);
    }
};

#endif
