#ifndef LEARN_ASSIGNMENT1_PAGE_RANK_SEQUENTIAL_H
#define LEARN_ASSIGNMENT1_PAGE_RANK_SEQUENTIAL_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

class page_rank_sequential{
    const string &filename;
    vector<vector<float>> graph, contributions;
    vector<float> z_order, rank;
    static auto interleave(u_int16_t, u_int16_t);
	float out_degree(int);
public:
    explicit page_rank_sequential(const string &);
	vector<float> compute_page_rank(float);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_SEQUENTIAL_H
