#ifndef N3LDG_ADDOP_H
#define N3LDG_ADDOP_H

class PAddNode : public Node {
  public:
	  PAddNode() {
		  this->node_type = "add";
	}
	~PAddNode() {
		clearValue();
	}

    virtual void clearValue();
    void forward(Graph *graph, const vector<PNode> &inputNodes);
    void backward();

    void compute();
    PExecute generate() override;

    vector<PNode> _inputNodes;
};

void PAddNode::clearValue() {
  _inputNodes.clear();
  Node::clearValue();
}

void PAddNode::backward() {
  for (Node *node : _inputNodes){
   node->loss.vec() += loss.vec();
  }

  if (this->_backword_callback_function != NULL) {
	  _backword_callback_function(loss.vec());
  }
}

void PAddNode::forward(Graph * graph, const vector<PNode> &inputNodes) {
  assert(_inputNodes.empty());

  degree = inputNodes.size();
  loss = 0;

  for (PNode pNode : inputNodes) {
    _inputNodes.push_back(pNode);
    pNode->parents.push_back(this);
  }

  graph->addNode(this);
}

void PAddNode::compute() {
  for (Node* pnode : _inputNodes) {
    val.vec() += pnode->val.vec();
  }
}

class AddExecutor : public Execute {
  public:
    void forward() override;
    void backward() override;
};

void AddExecutor::forward() {
  for (Node *node : batch) {
    static_cast<PAddNode *>(node)->compute();
  }
}

void AddExecutor::backward() {
  for (Node *node : batch) {
    static_cast<PAddNode *>(node)->backward();
  }
}

PExecute PAddNode::generate() {
  PExecute executor = new AddExecutor();
  executor->batch.push_back(this);
  return  executor;
}

#endif
