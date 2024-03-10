#ifndef LEARN_ASSIGNMENT1_PAGE_RANK_SEQUENTIAL_H
#define LEARN_ASSIGNMENT1_PAGE_RANK_SEQUENTIAL_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

class page_rank_sequential {
    const string &filename;
    vector<vector<int>> matrix;
    auto interleave(u_int16_t x, u_int16_t y) const;
    void parse_dataset();
public:
    explicit page_rank_sequential(const string &filename);
};

#endif //LEARN_ASSIGNMENT1_PAGE_RANK_SEQUENTIAL_H
