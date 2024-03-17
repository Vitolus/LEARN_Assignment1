#include "page_rank.h"

auto page_rank::interleave(u_int16_t col, u_int16_t row){
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

double page_rank::out_degree(vector<vector<double>> &graph, int col){
	double degree = 0.0;
	for(const auto &row : graph){
		degree += row[col];
	}
	return degree;
}

page_rank::page_rank(const string &filename) : filename(filename){
	/// header processing
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
	this->dim = stoi(n_nodes);

	/// initialize graph with 1 where there is an edge
	vector<vector<double>> graph(this->dim, vector<double>(this->dim, 0.0));
	while(getline(file, line)){
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		graph[arrive][start] = 1;
	}
	file.close();

	/// initialize z_order with 1/out_degree or 1/dim
	this->z_order.resize(this->dim * this->dim, 0.0);
	vector<double> oj(this->dim, 0.0);
	for(int i =0; i< this->dim; ++i){
		oj[i] = out_degree(graph, i);
	}
	for(int i = 0; i < this->dim; ++i){
		for(int j = 0; j < this->dim; ++j){
			if(graph[i][j] == 1){
				this->z_order[interleave(j, i)] = 1.0/oj[j];
			}
			else if(oj[j] == 0){
				this->z_order[interleave(j, i)] = 1.0/this->dim;
			}
		}
	}
	this->rank.resize(this->dim, 1.0/this->dim);
}

vector<double> page_rank::compute_page_rank(int iter, double beta){
	vector<double> results(this->dim, 0.0);
	double c = (1.0 - beta) / this->dim;
	for(int k = 0; k < iter ; ++k){
		for(auto i = 0; i < this->dim; ++i){
			for(auto j = 0; j < this->dim; ++j){
				results[i] += beta * this->z_order[interleave(j, i)] * this->rank[j];
			}
			results[i] += c;
		}
		this->rank = results;
	}
	return this->rank;
}
