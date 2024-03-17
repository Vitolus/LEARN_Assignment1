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
    vector<double> z_order, rank;
	vector<pair<u_int16_t, u_int16_t>> z_order_pos;
    static auto interleave(u_int16_t, u_int16_t);
	static void deinterleave(u_int32_t, u_int16_t &, u_int16_t &);
	static double out_degree(vector<vector<double>> &, int);
public:
    explicit page_rank_sequential(const string &);
	vector<double> compute_page_rank(double);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_SEQUENTIAL_H
