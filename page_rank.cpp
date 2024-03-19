#include "page_rank.h"

template<typename T>
T page_rank::interleave(T col, T row){
	T answer = 0;
	for (size_t i = 0; i < sizeof(T) * 8; ++i) {
		answer |= ((col & (T(1) << i)) << i) | ((row & (T(1) << i)) << (i + 1));
	}
	return answer;
}

template<typename T>
void page_rank::deinterleave(T z, T &col, T &row) {
	col = row = 0;
	for (size_t i = 0; i < sizeof(T) * 8; ++i) {
		col |= (z & (T(1) << (2 * i))) >> i;
		row |= (z & (T(1) << (2 * i + 1))) >> (i + 1);
	}
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
//TODO: z = interleave(j, i); z_order[z] access memory not belonging to the vector
	cout << "start init z_order" << endl;
	for(uint i = 0; i < this->dim; ++i){
		for(uint j = 0; j < this->dim; ++j){
			auto z = interleave(j, i);
			if(z >= this->dim*this->dim){
				cout << "i: " << interleave(j, i) << " row: " << i << " col: " << j << endl;
				throw out_of_range("z is greater than dim*dim");
			}
			if(graph[i][j] == 1){
				float val = 1.0/oj[j];
				this->matrix[i][j] = val;
				this->z_order[z] = val;
			}
			else if(oj[j] == 0){
				float val = 1.0/this->dim;
				this->matrix[i][j] = val;
				this->z_order[z] = val;
			}
			if(this->z_order[z] != this->matrix[i][j]){
				cout << "i: " << interleave(j, i) << " row: " << i << " col: " << j << endl;
				cout << "z_order: " << this->z_order[i] << " matrix: " << this->matrix[i][j] << endl;
				throw runtime_error("z_order and matrix are not equal");
			}
		}
	}
	this->rank.resize(this->dim, 1.0/this->dim);
}

vector<float> page_rank::compute_page_rank(int iter, float beta){
	float c = (1.0 - beta) / this->dim;
	for(auto k = 0; k < iter; ++k){
		vector<float> results(this->dim, 0.0);
		cout << "iter: " << k << endl;
/*
		/// matrix approach
		for(auto i = 0; i < this->dim; ++i){
			float sum = 0.0;
			for(auto j = 0; j < this->dim; ++j){
//				sum += this->z_order[interleave(j, i)] * this->rank[j];
				sum += this->matrix[i][j] * this->rank[j];
			}
			results[i] = beta * sum + c;
		}
*/
		/// z_order approach
		for(uint i = 0; i < this->dim*this->dim; ++i){
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
