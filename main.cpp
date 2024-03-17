#include <vector>
#include <iostream>
#include <string>

using namespace std;

double out_degree(vector<vector<double>>& M, int j) {
	double out_degree = 0;
	for(const auto& row : M) {
		out_degree += row[j];
	}
	return out_degree;
}

uint64_t interleave(uint32_t x, uint32_t y) {
	static const uint64_t M[] = {0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF, 0x0000FFFF0000FFFF};
	static const unsigned int S[] = {1, 2, 4, 8, 16};
	x = (x | (x << S[4])) & M[4];
	x = (x | (x << S[3])) & M[3];
	x = (x | (x << S[2])) & M[2];
	x = (x | (x << S[1])) & M[1];
	x = (x | (x << S[0])) & M[0];
	y = (y | (y << S[4])) & M[4];
	y = (y | (y << S[3])) & M[3];
	y = (y | (y << S[2])) & M[2];
	y = (y | (y << S[1])) & M[1];
	y = (y | (y << S[0])) & M[0];
	return x | (y << 1);
}

int main() {
	// Create an adjacency matrix
	vector<vector<double>> M = {
			{0, 0, 0, 0},
			{1, 0, 1, 1},
			{0, 0, 0, 0},
			{1, 0, 1, 0}
	};
	vector<vector<double>> Z = {
			{0,0,0,0},
			{0,0,0,0},
			{0,0,0,0},
			{0,0,0,0}
	};

	vector<double> oj(M.size(), 0.0);
	vector<double> z_order(M.size()*M.size(), 0.0);
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
}