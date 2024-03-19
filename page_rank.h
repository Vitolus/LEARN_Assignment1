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
    vector<float> z_order, rank;
	vector<vector<float>> matrix;
	uint dim;
	template<typename T>
	T interleave(T col, T row);
	template<typename T>
	static void deinterleave(T, T &, T &);
	static float out_degree(vector<vector<float>> &, int);
public:
    explicit page_rank(const string &);
	vector<float> compute_page_rank(int, float);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_H
