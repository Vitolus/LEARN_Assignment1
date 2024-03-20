#include "page_rank.h"

// Function to calculate outdegree of each node
vector<float> page_rank::outDegree(const vector<vector<short>>& graph) {
	vector<float> outdegree(graph.size(), 0);
	for (int j = 0; j < graph.size(); ++j) {
		for (const auto & i : graph) {
			outdegree[j] += i[j];
		}
	}
	return outdegree;
}

page_rank::page_rank(const string &filename) : filename(filename), vals({0.0}), cols(), rows(){
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
	vector<vector<short>> graph(this->dim, vector<short>(this->dim, 0.0));
	while(getline(file, line)){
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		graph[arrive][start] = 1;
	}
	file.close();

//TODO: take times
	/// csr approach
	vector<float> oj = outDegree(graph);
	for(int i = 0; i < this->dim; ++i){
		for(int j = 0; j < this->dim; ++j){
			if(graph[i][j] == 1){
				this->vals.push_back(1.0 / oj[j]);
				this->cols.push_back(j);
			}else if(oj[j] == 0){
				this->vals.push_back(1.0 / this->dim);
				this->cols.push_back(j);
			}
		}
		this->rows.push_back(this->vals.size());
	}

	/// matrix approach
	/*
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
			}
			else if(oj[j] == 0){
				float val = 1.0/this->dim;
				this->matrix[i][j] = val;
			}
		}
	}
	*/
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
