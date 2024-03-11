#include "page_rank_sequential.h"

auto page_rank_sequential::interleave(u_int16_t x, u_int16_t y){
	static const u_int16_t M[] = {0x5555, 0x3333, 0x0F0F, 0x00FF};
	static const u_int16_t S[] = {1, 2, 4, 8};
	x = (x | (x << S[3])) & M[3];
	x = (x | (x << S[2])) & M[2];
	x = (x | (x << S[1])) & M[1];
	x = (x | (x << S[0])) & M[0];
	y = (y | (y << S[3])) & M[3];
	y = (y | (y << S[2])) & M[2];
	y = (y | (y << S[1])) & M[1];
	y = (y | (y << S[0])) & M[0];
	auto result = x | (y << 1);
	return result;
}

page_rank_sequential::page_rank_sequential(const string &filename) : filename(filename){
	ifstream file(filename);
	if(!file.is_open()){
		cout << "file not found" << endl;
		return;
	}
	string line;
	for(int i = 0; i < 2; ++i){
		getline(file, line); // skip first two lines
	}
	getline(file, line); // line with number of nodes and edges
	istringstream iss(line);
	string n_nodes;
	while(iss >> n_nodes){
		if(n_nodes == "Nodes:"){
			iss >> n_nodes;
			break;
		}
	}
	getline(file, line); // skip last header line
	uint dim = stoi(n_nodes);
	matrix.resize(dim, vector<int>(dim, 0));
	z_order.resize(dim * dim, 0);
	while(getline(file, line)){
		istringstream iss(line);
		int row, col;
		iss >> col >> row;
		matrix[row][col] = 1;
		z_order[interleave(row, col)] = 1;
	}
	file.close();
}
