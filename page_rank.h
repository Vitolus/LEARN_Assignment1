#ifndef LEARN_ASSIGNMENT1_PAGE_RANK_H
#define LEARN_ASSIGNMENT1_PAGE_RANK_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <omp.h>

using namespace std;

class page_rank{
    const string &filename;
    vector<float>  rank;
	vector<vector<float>> matrix;
	unordered_map<uint, float> z_order;
	uint dim;
	static uint interleave(uint, uint);
	static pair<uint, uint> deinterleave(uint);
	static float out_degree(vector<vector<float>> &, int);
public:
    explicit page_rank(const string &);
	vector<float> compute_page_rank(int, float);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_H
