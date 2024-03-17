#ifndef LEARN_ASSIGNMENT1_PAGE_RANK_H
#define LEARN_ASSIGNMENT1_PAGE_RANK_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

class page_rank{
    const string &filename;
    vector<double> z_order, rank;
	uint dim;
    static auto interleave(u_int16_t, u_int16_t);
	static double out_degree(vector<vector<double>> &, int);
public:
    explicit page_rank(const string &);
	vector<double> compute_page_rank(int, double);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_H
