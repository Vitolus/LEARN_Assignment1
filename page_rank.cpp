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
	dim = stoi(n_nodes);
	/// initialize graph with 1 where there is an edge
	vector<vector<short>> graph(dim, vector<short>(dim, 0));
	while(getline(file, line)){
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		graph[arrive][start] = 1;
	}
	file.close();

	cout << "start init oj" << endl;
	vector<int> oj = outDegree(graph);
	rows = {0};
	cout << "start init transition" << endl;
	for(auto i = 0; i < dim; ++i){
		for(auto j = 0; j < dim; ++j){
			if(graph[i][j] == 1){
				auto val = static_cast<float>(1.0/oj[j]);
				vals.push_back(val);
				cols.push_back(j);
			}
			else if(oj[j] == 0){
				auto val = static_cast<float>(1.0/dim);
				vals.push_back(val);
				cols.push_back(j);
			}
		}
		rows.push_back(vals.size());
	}
	rank.resize(dim, static_cast<float>(1.0/dim));
}

const vector<float> &page_rank::getRank() const{
	return rank;
}

vector<float> page_rank::compute_page_rank(int iter, float beta){
	rank.resize(dim, static_cast<float>(1.0/dim));
	auto c = static_cast<float>((1.0 - beta) / dim);
	for(auto k = 0; k < iter; ++k){
		vector<float> results(dim, 0.0);
		cout << "iter: " << k << endl;
		for(auto i = 0; i < dim; ++i){
			float sum = 0.0;
			for(auto j = rows[i]; j < rows[i+1]; ++j){
				sum += vals[j] * rank[cols[j]];
			}
			results[i] = beta * sum + c;
		}
		rank = results;
	}
	return rank;
}
