#include "page_rank.h"

uint page_rank::interleave(uint x, uint y) {
	uint z = 0;  // Initialize result

	// Assuming 32-bit integers
	for (auto i = 0; i < sizeof(x) * 8; i++) {
		z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
	}
	return z;
}


pair<uint, uint> page_rank::deinterleave(uint z) {
	uint x = 0, y = 0;  // Initialize result

	// Assuming 32-bit integers
	for (auto i = 0; i < sizeof(z) * 8; i += 2) {
		x |= ((z & (1 << i)) >> i);
		y |= ((z & (1 << (i + 1))) >> i);
	}
	return std::make_pair(x, y);
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
	this->matrix.resize(this->dim, vector<float>(this->dim, 0.0));

	cout << "start init oj" << endl;
	vector<float> oj(this->dim, 0.0);
	for(int i =0; i< this->dim; ++i){
		oj[i] = out_degree(graph, i);
	}

//TODO: z = interleave(j, i); z_order[z] access memory not belonging to the vector
	cout << "start init z_order" << endl;
	for(auto i = 0; i < this->dim; ++i){
		for(auto j = 0; j < this->dim; ++j){
			auto z = interleave(j, i);
			if(z >= this->dim*this->dim){
				cout << "z: " << interleave(j, i) << " row: " << i << " col: " << j << endl;
				throw out_of_range("z is greater than dim*dim");
			}
			if(graph[i][j] == 1){
				float val = 1.0/oj[j];
				this->matrix[i][j] = val;
				this->z_order[z] = val;
				try{
					if(this->z_order.at(z) != this->matrix[i][j]){
						cout << "z: " << interleave(j, i) << " row: " << i << " col: " << j << endl;
						cout << "z_order: " << this->z_order.at(z) << " matrix: " << this->matrix[i][j] << endl;
						throw runtime_error("z_order and matrix are not equal");
					}
				}catch(const out_of_range &e){
					cout << "key not found: " << z << " , " << interleave(j, i) << " row: " << i << " col: " << j << endl;
				}
			}
			else if(oj[j] == 0){
				float val = 1.0/this->dim;
				this->matrix[i][j] = val;
				this->z_order[z] = val;
				try{
					if(this->z_order.at(z) != this->matrix[i][j]){
						cout << "z: " << interleave(j, i) << " row: " << i << " col: " << j << endl;
						cout << "z_order: " << this->z_order.at(z) << " matrix: " << this->matrix[i][j] << endl;
						throw runtime_error("z_order and matrix are not equal");
					}
				}catch(const out_of_range &e){
					cout << "key not found: " << z << " , " << interleave(j, i) << " row: " << i << " col: " << j << endl;
				}
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

		/// matrix approach
		for(auto i = 0; i < this->dim; ++i){
			float sum = 0.0;
			for(auto j = 0; j < this->dim; ++j){
//				sum += this->z_order[interleave(j, i)] * this->rank[j];
				sum += this->matrix[i][j] * this->rank[j];
			}
			results[i] = beta * sum + c;
		}

/*
		/// z_order approach
		for(u_int32_t i = 0; i < this->dim*this->dim; ++i){
			u_int16_t col, row;
			deinterleave(i, col, row);
			results[row] += this->z_order[i] * this->rank[col];
		}
		for(auto i = 0; i < this->dim; ++i){
			results[i] = beta * results[i] + c;
		}
*/
		this->rank = results;
	}
	return this->rank;
}
