#ifndef N3LDG_ATTENTION_H
#define N3LDG_ATTENTION_H

#include <vector>
#include <memory>

#include "PAddOP.h"
#include "BiOP.h"
#include "UniOP.h"
#include "ScalarMultiOP.h"
#include "Graph.h"
#include "ActivatedNode.h"

class AttentionBuilder {
public:
    ScalarMultiNode _context;
    PAddNode _context_numerator;

    std::vector<BiNode> _unprobalized_weight_nodes;
    std::vector<ActivatedNode> _exp_nodes;
    std::vector<ScalarMultiNode> _weighted_hidden_nodes;

    void forward(Graph *graph, const std::vector<PNode> &encoder_hiddens, const PNode previous_decoder_hidden) {
        for (PNode encoder_hidden : encoder_hiddens) {
            _unprobalized_weight_nodes.resize(_unprobalized_weight_nodes.size() + 1);
            BiNode *binode = &_unprobalized_weight_nodes.at(_unprobalized_weight_nodes.size() - 1);
            binode->forward(graph, previous_decoder_hidden, encoder_hidden);

            _exp_nodes.resize(_exp_nodes.size() + 1);
            ActivatedNode *activatednode = &_exp_nodes.at(_exp_nodes.size() - 1);
            activatednode->forward(graph, binode);

            _weighted_hidden_nodes.resize(_weighted_hidden_nodes.size() + 1);
            ScalarMultiNode *scalarmultinode = &_weighted_hidden_nodes.at(_weighted_hidden_nodes.size() - 1);
            scalarmultinode->forward(graph, activatednode, encoder_hidden);
        }

        std::vector<PNode> ptrs = toPointers<ScalarMultiNode, Node>(_weighted_hidden_nodes);
    }
};

#endif