#include <vector>
#include <unordered_map>
#include <iostream>
#include "page_rank.h"

using namespace std;

float out_degree(vector<vector<float>>& M, int j) {
	float out_degree = 0;
	for(const auto& row : M) {
		out_degree += row[j];
	}
	return out_degree;
}

uint interleave(uint x, uint y) {
	uint z = 0;  // Initialize result

	// Assuming 32-bit integers
	for (uint i = 0; i < sizeof(x) * 8; i++) {
		z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
	}

	return z;
}


int main() {
	auto *pr = new page_rank("../datasets/p2p-Gnutella25.txt");
	auto rank = pr->compute_page_rank(50, 0.85);
	for(auto &i : rank){
		cout << i << endl;
	}
	return 0;
}