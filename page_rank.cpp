#include "page_rank.h"

uint page_rank::interleave(uint col, uint row){
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

void page_rank::deinterleave(uint z, uint &col, uint &row) {
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

float page_rank::out_degree(vector<vector<float>> &graph, int col){
	float degree = 0.0;
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
	vector<vector<float>> graph(this->dim, vector<float>(this->dim, 0.0));
	while(getline(file, line)){
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		graph[arrive][start] = 1;
	}
	file.close();
//TODO: take times
	cout << "start init" << endl;
	this->z_order.resize(this->dim * this->dim, 0.0);
	this->matrix.resize(this->dim, vector<float>(this->dim, 0.0));
	vector<float> oj(this->dim, 0.0);
	cout << "start init oj" << endl;
	for(int i =0; i< this->dim; ++i){
		oj[i] = out_degree(graph, i);
	}

	/// matrix approach
	cout << "start init matrix" << endl;
	for(int i = 0; i < this->dim; ++i){
		for(int j = 0; j < this->dim; ++j){
			if(graph[i][j] == 1){
				this->matrix[i][j] = 1.0/oj[j];
			}
			else if(oj[j] == 0){
				this->matrix[i][j] = 1.0/this->dim;
			}
		}
	}

	/// z_order approach
	cout << "start init z_order" << endl;
	for(int i = 0; i< this->dim*this->dim; ++i){
		uint col, row;
		deinterleave(i, col, row);
		/*
		if(graph[row][col] == 1){
			this->z_order[i] = 1.0/oj[col];
		}
		else if(oj[col] == 0){
			this->z_order[i] = 1.0/this->dim;
		}
		 */
		this->z_order[i] = matrix[row][col];

	}

	this->rank.resize(this->dim, 1.0/this->dim);
}

vector<float> page_rank::compute_page_rank(int iter, float beta){
	float c = (1.0 - beta) / this->dim;
	vector<float> results(this->dim, 0.0);
	for(auto k = 0; k < iter; ++k){
		cout << "iter: " << k << endl;

		/// matrix approach
		for(auto i = 0; i < this->dim; ++i){
			float sum = 0.0;
			for(auto j = 0; j < this->dim; ++j){
//				sum += this->z_order[interleave(j, i)] * this->rank[j];
				sum += this->matrix[i][j] * this->rank[j];
			}
			results[i] = beta * sum + c;
		}

		//TODO: z_order approach rotto
		for(auto i = 0; i < this->dim*this->dim; ++i){
			uint col, row;
			deinterleave(i, col, row);
			results[row] += this->z_order[i] * this->rank[col];
		}
		for(auto i = 0; i < this->dim; ++i){
			results[i] = beta * results[i] + c;
		}

		this->rank = results;
	}
	return this->rank;
}
