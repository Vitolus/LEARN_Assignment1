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
	uint dim;
    vector<float> rank;
	vector<uint> rows, cols;
	vector<float> vals;
	vector<vector<float>> matrix;
	static float out_degree(vector<vector<float>> &, int);
public:
    explicit page_rank(const string &);
	vector<float> compute_page_rank(int, float);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_H
