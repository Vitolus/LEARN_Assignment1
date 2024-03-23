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
	int dim;
    vector<float> rank;
	vector<uint> rows, cols;
	vector<float> vals;
	vector<vector<float>> matrix;
	static vector<int> outDegree(const vector<vector<short>> &);
public:
    explicit page_rank(const string &);
	[[nodiscard]] const vector<float> &getRank() const;
	void compute_page_rank(int, int, float);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_H
