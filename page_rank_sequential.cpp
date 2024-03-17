#include "page_rank_sequential.h"

auto page_rank_sequential::interleave(u_int16_t col, u_int16_t row){
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

void page_rank_sequential::deinterleave(u_int32_t z, u_int16_t &col, u_int16_t &row) {
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

double page_rank_sequential::out_degree(vector<vector<double>> &graph, int col){
	double degree = 0.0;
	for(const auto &row : graph){
		degree += row[col];
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
		getline(file, line); // skip first two header lines
	}
	getline(file, line); // line with number of nodes and edges
	istringstream iss(line);
	string n_nodes;
	while(iss >> n_nodes){
		if(n_nodes == "Nodes:"){
			iss >> n_nodes;  // get number of nodes
			break;
		}
	}
	getline(file, line); // skip last header line
	uint dim = stoi(n_nodes);

	/// initialize graph with 1 where there is an edge
	vector<vector<double>> graph(dim, vector<double>(dim, 0.0));
	while(getline(file, line)){
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		graph[arrive][start] = 1;
	}
	file.close();

	rank.resize(dim, 1.0/dim);
	/// initialize z_order with 1/out_degree or 1/dim
	z_order.resize(dim*dim, 0.0);
	for(int i = 0; i < dim; ++i){
		for(int j = 0; j < dim; ++j){
			double oj = out_degree(graph, j);
			if(graph[i][j] == 1){
				z_order[interleave(j, i)] = 1.0/oj;
			}
			else if(oj == 0){
				z_order[interleave(j, i)] = 1.0/dim;
			}
		}
	}
}

vector<double> page_rank_sequential::compute_page_rank(double beta){
	uint dim = rank.size();
	vector<double> results(dim, 0.0);
	double c = (1.0 - beta) / dim;
	for(int i = 0; i < dim; ++i){
		for(int j = 0; j < dim; ++j){
			results[i] += beta * z_order[interleave(i, j)] * rank[j];
		}
		results[i] += c;
	}
	rank = results;
	return rank;
}
