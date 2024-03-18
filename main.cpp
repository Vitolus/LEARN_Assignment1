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

auto interleave(u_int16_t col, u_int16_t row){
	static const u_int16_t M[] = {0x5555, 0x3333, 0x0F0F, 0x00FF};
	static const u_int16_t S[] = {1, 2, 4, 8};
	col = (col | (col << S[3])) & M[3];
	col = (col | (col << S[2])) & M[2];
	col = (col | (col << S[1])) & M[1];
	col = (col | (col << S[0])) & M[0];
	row = (row | (row << S[3])) & M[3];
	row = (row | (row << S[2])) & M[2];
	row = (row | (row << S[1])) & M[1];
	row = (row | (row << S[0])) & M[0];
	auto result = col | (row << 1);
	return result;
}

void deinterleave(u_int32_t z, u_int16_t &col, u_int16_t &row) {
	static const u_int16_t M[] = {0x5555, 0x3333, 0x0F0F, 0x00FF};
	static const u_int16_t S[] = {0, 1, 2, 4};
	col = z;
	row = z >> 1;
	col = (col | (col >> S[3])) & M[3];
	col = (col | (col >> S[2])) & M[2];
	col = (col | (col >> S[1])) & M[1];
	col = (col | (col >> S[0])) & M[0];
	row = (row | (row >> S[3])) & M[3];
	row = (row | (row >> S[2])) & M[2];
	row = (row | (row >> S[1])) & M[1];
	row = (row | (row >> S[0])) & M[0];
}


vector<float> compute_page_rank(int iter, float beta, int dim, vector<float>& z_order, vector<float>& rank){
	float c = (1.0 - beta) / dim;
	for(int k = 0; k < iter ; ++k){
		vector<float> results(dim, 0.0);
		for(auto i = 0; i < dim; ++i){
			for(auto j = 0; j < dim; ++j){
				results[i] += beta * z_order[interleave(j, i)] * rank[j];
			}
			results[i] += c;
		}
		rank = results;
	}
	return rank;
}


int main() {
	 for(short i = 0; i < 4; ++i){
		 for(short j = 0; j < 4; ++j){
			 cout << interleave(j, i) << " ";
		 }
		 cout << endl;
	 }
	 for(int i = 0; i < 16; ++i){
		 u_int16_t col, row;
		 deinterleave(i, col, row);
		 cout << row << " " << col << endl;
	 }

	/*
	page_rank pr("../datasets/p2p-Gnutella25.txt");
	auto rank = pr.compute_page_rank(50, 0.85);
	for(auto &i : rank){
		cout << i << endl;
	}
	 */
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
				Z[i][j] = 1.0/oj[j];
				z_order[interleave(j, i)] = Z[i][j];
			}
			else if(oj[j] == 0){
				Z[i][j] = 1.0/M.size();
				z_order[interleave(j, i)] = Z[i][j];
			}
		}
	}
	vector<float> rank(M.size(), 1.0/M.size());
	compute_page_rank(50, 0.85, M.size(), z_order, rank);
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

	return 0;
 */
}