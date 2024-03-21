#include "page_rank.h"

// Function to calculate outdegree of each node
vector<int> page_rank::outDegree(const vector<vector<short>>& graph) {
	vector<int> oj(graph.size(), 0);
	for (auto j = 0; j < graph[0].size(); ++j) {
		for (const auto & i : graph) {
			oj[j] += i[j];
		}
	}
	return oj;
}

page_rank::page_rank(const string &filename, int n_thread) : filename(filename), n_threads(n_thread){
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
	vector<vector<short>> graph(this->dim, vector<short>(this->dim, 0));
	while(getline(file, line)){
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		graph[arrive][start] = 1;
	}
	file.close();

	cout << "start init oj" << endl;
	vector<int> oj = outDegree(graph);
	this->rows = {0};
	cout << "start init transition" << endl;
	for(auto i = 0; i < this->dim; ++i){
		for(auto j = 0; j < this->dim; ++j){
			if(graph[i][j] == 1){
				auto val = static_cast<float>(1.0/oj[j]);
				this->vals.push_back(val);
				this->cols.push_back(j);
			}
			else if(oj[j] == 0){
				auto val = static_cast<float>(1.0/this->dim);
				this->vals.push_back(val);
				this->cols.push_back(j);
			}
		}
		this->rows.push_back(this->vals.size());
	}
	this->rank.resize(this->dim, static_cast<float>(1.0/this->dim));
}

vector<float> page_rank::compute_page_rank(int iter, float beta){
	this->rank.resize(this->dim, static_cast<float>(1.0/this->dim));
	auto c = static_cast<float>((1.0 - beta) / this->dim);
	for(auto k = 0; k < iter; ++k){
		vector<float> results(this->dim, 0.0);
		cout << "iter: " << k << endl;
		for(auto i = 0; i < this->dim; ++i){
			float sum = 0.0;
			for(auto j = this->rows[i]; j < this->rows[i+1]; ++j){
				sum += this->vals[j] * this->rank[this->cols[j]];
			}
			results[i] = beta * sum + c;
		}
		this->rank = results;
	}
	return this->rank;
}
