#ifndef N3LDG_TANH_OP_H
#define N3LDG_TANH_OP_H

struct TanhNode : Node {
  PNode in;

  TanhNode() : in(NULL) {
	  node_type = "tanh";
  }

  ~TanhNode() {
	  clearValue();
  }

  virtual void clearValue() {
    in = NULL;
    Node::clearValue();
  }

  void forward(Graph * graph, PNode inNode) {
    in = inNode;
    degree = 1;
    loss = 0;
	inNode->parents.push_back(this);
    graph->addNode(this);
  }

  void compute() {
    val.vec() = in->val.vec().unaryExpr(ptr_fun(ftanh));
  }

  void backward() {
    in->loss.vec() += loss.vec() * in->val.vec().binaryExpr(val.vec(), ptr_fun(dtanh));
  }

  PExecute generate() override;
};

class TanhExecutor : public Execute {
  public:
    void forward() override;
    void backward() override;
};

void TanhExecutor::forward() {
  for (Node *node : batch) {
    static_cast<TanhNode *>(node)->compute();
  }
}

void TanhExecutor::backward() {
  for (Node *node : batch) {
    static_cast<TanhNode *>(node)->backward();
  }
}

PExecute TanhNode::generate() {
  PExecute executor = new TanhExecutor();
  executor->batch.push_back(this);
  return executor;
}

#endif
