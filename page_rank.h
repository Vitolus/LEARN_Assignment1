#ifndef LEARN_ASSIGNMENT1_PAGE_RANK_H
#define LEARN_ASSIGNMENT1_PAGE_RANK_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <fstream>
#include <sstream>
#include <omp.h>

using namespace std;

class page_rank{
    const string &filename;
	long nodes;
	long edges;
	vector<ulong> *rows, *cols;
	vector<float> *vals;
	vector<float> *rank;
	[[nodiscard]] vector<ulong> *outDegree(const unordered_map<ulong, unordered_set<ulong>>*) const;
public:
    explicit page_rank(const string &);
	[[nodiscard]] long getNodes() const;
	[[nodiscard]] long getEdges() const;
	vector<float> *compute_page_rank(int, int, float);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_H
