#include "page_rank.h"

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
	vector<vector<float>> graph(this->dim, vector<float>(this->dim, 0.0));
	/// initialize graph with 1 where there is an edge
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

	cout << "start init csr" << endl;
	for(auto i = 0; i < this->dim; ++i){
		for(auto j = 0; j < this->dim; ++j){
			if(graph[i][j] == 1){
				float val = 1.0/oj[j];
				this->matrix[i][j] = val;
				this->rows.push_back(i);
				this->cols.push_back(j);
				this->vals.push_back(val);
			}
			else if(oj[j] == 0){
				float val = 1.0/this->dim;
				this->matrix[i][j] = val;
				this->rows.push_back(i);
				this->cols.push_back(j);
				this->vals.push_back(val);
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
				sum += this->matrix[i][j] * this->rank[j];
			}
			results[i] = beta * sum + c;
		}

		/// csr approach
//TODO: csr approach
		this->rank = results;
	}
	return this->rank;
}
