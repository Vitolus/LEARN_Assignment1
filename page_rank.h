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
	int nodes;
	int edges;
    vector<float> rank;
	vector<uint> rows, cols;
	vector<float> vals;
	vector<vector<float>> matrix;
	[[nodiscard]] vector<int> outDegree(const vector<vector<short>>&) const;
	[[nodiscard]] vector<int> outDegree(const unordered_map<int, unordered_set<int>>&) const;
public:
    explicit page_rank(const string &);
	[[nodiscard]] const vector<float> &getRank() const;
	[[nodiscard]] int getNodes() const;
	[[nodiscard]] int getEdges() const;
	float compute_page_rank(int, int, float);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_H
