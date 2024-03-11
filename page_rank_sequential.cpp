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

float page_rank_sequential::out_degree(int row){
	float degree = 0;
	for(auto &col : graph[row]){
		if(col != 0){
			++degree;
		}
	}
	return degree;
}

page_rank_sequential::page_rank_sequential(const string &filename) : filename(filename){
	ifstream file(this->filename);
	if(!file.is_open()){
		cout << "file not found" << endl;
		return;
	}
	string line;
	for(short i = 0; i < 2; ++i){
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
	graph.resize(dim, vector<float>(dim, 0.0));
	while(getline(file, line)){
		iss.str(line);
		int row, col;
		iss >> row >> col;
		graph[row][col] = 1;
	}
	file.close();
	contributions.resize(dim, vector<float>(dim, 0.0));
	z_order.resize(dim * dim, 0.0);
	rank.resize(dim, 1.0/dim);
	for(int i = 0; i < dim; ++i){
		for(int j = 0; j < dim; ++j){
			float oj = out_degree(j);
			if(graph[j][i] == 1){
				contributions[i][j] = 1.0/oj;
				z_order[interleave(i, j)] = 1.0/oj;
			}else if(oj == 0){
				contributions[i][j] = 1.0/dim;
				z_order[interleave(i, j)] = 1.0/dim;
			}
		}
	}
}

vector<float> page_rank_sequential::compute_page_rank(float beta){
	uint dim = rank.size();
	vector<float> results(dim, 0.0);
	float c = (1.0 - beta) / dim;
	for(int i = 0; i < dim; ++i){
		for(int j = 0; j < dim; ++j){
			results[i] += beta * z_order[interleave(i, j)] * rank[j];
		}
		results[i] += c;
	}
	rank = results;
	return rank;
}
