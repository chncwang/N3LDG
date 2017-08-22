#ifndef N3LDG_ATTENTION_H
#define N3LDG_ATTENTION_H

#include <vector>

#include "PAddOP.h"
#include "ScalarMultiOP.h"

class AttentionBuilder {
public:
  std::vector<PAddNode> _contexts;

};

#endif