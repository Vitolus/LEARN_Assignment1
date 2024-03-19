#include <vector>
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

uint interleave(uint col, uint row){
	static const uint M[] = {0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF};
	static const uint S[] = {1, 2, 4, 8};
	col = (col | (col << S[3])) & M[3];
	col = (col | (col << S[2])) & M[2];
	col = (col | (col << S[1])) & M[1];
	col = (col | (col << S[0])) & M[0];
	row = (row | (row << S[3])) & M[3];
	row = (row | (row << S[2])) & M[2];
	row = (row | (row << S[1])) & M[1];
	row = (row | (row << S[0])) & M[0];
	return col | (row << 1);
}

void deinterleave(uint z, uint &col, uint &row) {
	static const uint M[] = {0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
	static const uint S[] = {1, 2, 4, 8};

	col = z & 0x55555555;
	row = (z >> 1) & 0x55555555;
	col = (col | (col >> S[0])) & M[0];
	col = (col | (col >> S[1])) & M[1];
	col = (col | (col >> S[2])) & M[2];
	col = (col | (col >> S[3])) & M[3];
	row = (row | (row >> S[0])) & M[0];
	row = (row | (row >> S[1])) & M[1];
	row = (row | (row >> S[2])) & M[2];
	row = (row | (row >> S[3])) & M[3];
}


void compute_page_rank(int iter, float beta, int dim, vector<vector<float>> &matrix, vector<float>& z_order, vector<float>& rank){
	float c = (1.0 - beta) / dim;
	for(auto k = 0; k < iter; ++k){
		cout << "iter: " << k << endl;
		vector<float> results(dim, 0.0);

		/// z_order approach
		for(auto i = 0; i < dim*dim; ++i){
			uint col, row;
			deinterleave(i, col, row);
			results[row] += z_order[i] *rank[col];
		}
		for(auto i = 0; i < dim; ++i){
			results[i] = beta * results[i] + c;
		}

		rank = results;
	}
}


int main() {
	auto *pr = new page_rank("../datasets/p2p-Gnutella25.txt");
	auto rank = pr->compute_page_rank(50, 0.85);
	for(auto &i : rank){
		cout << i << endl;
	}
/*
	// Create an adjacency matrix
	vector<vector<float>> M = {
			{0, 0, 0, 0},
			{1, 0, 1, 1},
			{0, 0, 0, 0},
			{1, 0, 1, 0}
	};
	vector<vector<float>> Z = {
			{0,0,0,0},
			{0,0,0,0},
			{0,0,0,0},
			{0,0,0,0}
	};

	vector<float> oj(M.size(), 0.0);
	vector<float> z_order(M.size()*M.size(), 0.0);
	for(int i =0; i< M.size(); ++i){
		oj[i] = out_degree(M, i);
	}
	for(int i = 0; i < M.size(); ++i){
		for(int j = 0; j < M.size(); ++j){
			if(M[i][j] == 1){
				z_order[interleave(j, i)] = 1.0/oj[j];
			}
			else if(oj[j] == 0){
				z_order[interleave(j, i)] = 1.0/M.size();
			}
		}
	}
	vector<float> rank(M.size(), 1.0/M.size());
	compute_page_rank(50, 0.85, M.size(), Z, z_order, rank);
	cout << "Rankm: ";
	for(auto &i : rank){
		cout << i << " ";
	}
	cout << endl;
	for(auto &i : oj){
		cout << i << " ";
	}
	cout << endl;
	for(auto &row : Z){
		for(auto &col : row){
			cout << col << "    ";
		}
		cout << endl;
	}
	for(auto &i : z_order){
		cout << i << " ";
	}
*/
	return 0;
}