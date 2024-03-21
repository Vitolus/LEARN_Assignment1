#ifndef LEARN_ASSIGNMENT1_PAGE_RANK_H
#define LEARN_ASSIGNMENT1_PAGE_RANK_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <omp.h>

using namespace std;

class page_rank{
    const string &filename;
	int n_threads;
	uint dim;
    vector<float> rank;
	vector<uint> rows, cols;
	vector<float> vals;
	vector<vector<float>> matrix;
	static vector<int> outDegree(const vector<vector<short>> &);
public:
    explicit page_rank(const string &, int);
	vector<float> compute_page_rank(int, float);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_H
