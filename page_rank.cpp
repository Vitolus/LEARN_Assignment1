#include "page_rank.h"

vector<int> page_rank::outDegree(const vector<vector<short>>& graph) {
	vector<int> oj(graph.size(), 0);
	for (auto j = 0; j < graph[0].size(); ++j) {
		for (const auto & i : graph) {
			oj[j] += i[j];
		}
	}
	return oj;
}

page_rank::page_rank(const string &filename) : filename(filename), dim(0){
	/// first pass to get the dimension of the graph
	cout << "first pass reading file" << endl;
	ifstream file(this->filename);
	if(!file.is_open()){
		cout << "file not found" << endl;
		return;
	}
	string line;
	istringstream iss;
	while(getline(file, line)){
		if(line[0] == '#'){
			continue;
		}
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		dim = max(dim, max(start, arrive) + 1);
		break;
	}
	while(getline(file, line)){
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		dim = max(dim, max(start, arrive) + 1);
	}
	file.close();
	cout << "dimension of the graph: " << dim << endl;
	vector<vector<short>> graph(dim, vector<short>(dim, 0));
	cout << "second pass reading file" << endl;
	/// second pass to initialize graph with 1 where there is an edge
	file.open(this->filename);
	if(!file.is_open()){
		cout << "file not found" << endl;
		return;
	}
	while(getline(file, line)){
		if(line[0] == '#'){
			continue;
		}
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		graph[arrive][start] = 1;
		break;
	}
	while(getline(file, line)){
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		graph[arrive][start] = 1;
	}
	file.close();
	/// compute out degree vector
	vector<int> oj = outDegree(graph);
	/// create CSR transition matrix
	rows = {0};
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
	/// initialize rank vector
	rank.resize(dim, static_cast<float>(1.0/dim));
}

const vector<float> &page_rank::getRank() const{
	return rank;
}

void page_rank::compute_page_rank(int n_threads, int iter, float beta){
	auto c = (1.0 - beta) / dim;
	for(auto k = 0; k < iter; ++k){
		vector<float> results(dim, 0.0);
		#pragma omp parallel for if(n_threads > 1) num_threads(n_threads) schedule(static) default(none) shared(beta, c, results)
		for(auto i = 0; i < dim; ++i){
			float sum = 0.0;
			for(auto j = rows[i]; j < rows[i+1]; ++j){
				sum += vals[j] * rank[cols[j]];
			}
			results[i] = static_cast<float>(beta * sum + c);
		}
		rank = results;
	}
}
