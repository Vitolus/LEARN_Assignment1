#include "page_rank.h"

vector<int> page_rank::outDegree(const unordered_map<int, unordered_set<int>>& graph_out) const{
	vector<int> oj(nodes, 0);
	for(auto i : graph_out){
		oj[i.first] = i.second.size();
	}
	return oj;
}

page_rank::page_rank(const string &filename) : filename(filename), nodes(0), edges(0){
	/// first pass to get the dimension of the graph
	cout << "first pass reading file" << endl;
	ifstream file(this->filename);
	if(!file.is_open()){
		cout << "file not found" << endl;
		return;
	}
	string line;
	while(getline(file, line)){
		if(line[0] == '#'){
			continue;
		}
		istringstream iss(line);
		int start, arrive;
		iss >> start >> arrive;
		nodes = max(nodes, max(start, arrive) + 1);
		++edges;
		break;
	}
	while(getline(file, line)){
		istringstream iss(line);
		int start, arrive;
		iss >> start >> arrive;
		nodes = max(nodes, max(start, arrive) + 1);
		++edges;
	}
	file.close();
	cout << "dimension of graph: " << nodes << endl;
	cout << "dimension of graph in ram: " << static_cast<float>(sizeof(short)*nodes*nodes)/(1024*1024*1024) << " GB" << endl;
	cout << "dimension of graph with only non zero values in ram: " << static_cast<float>(sizeof(short)*edges*2)/(1024*1024) << " MB" << endl;
	cout << "sparisity rate: " << static_cast<float>(edges)/(nodes*(nodes-1)) << endl;
	unordered_map<int, unordered_set<int>> graph_out(nodes);
	unordered_map<int, unordered_set<int>> graph_in(nodes);
	/// second pass to initialize graph with 1 where there is an edge
	cout << "second pass reading file" << endl;
	file.open(this->filename);
	if(!file.is_open()){
		cout << "file not found" << endl;
		return;
	}
	while(getline(file, line)){
		if(line[0] == '#'){
			continue;
		}
		istringstream iss(line);
		int start = 0, arrive = 0;
		iss >> start >> arrive;
		graph_out[start].insert(arrive);
		graph_in[arrive].insert(start);
		break;
	}
	while(getline(file, line)){
		istringstream iss(line);
		int start = 0, arrive = 0;
		iss >> start >> arrive;
		graph_out[start].insert(arrive);
		graph_in[arrive].insert(start);
	}
	file.close();
	/// compute out degree vector
	cout << "computing out degree vector" << endl;
	vector<int> oj2 = outDegree(graph_out);
	graph_out.clear(); // free memory
	/// create CSR transition matrix
	cout << "creating CSR transition matrix" << endl;
	rows = {0};
	for(auto i = 0; i < nodes; ++i){
		for(auto j = 0; j < nodes; ++j){
			if(graph_in.contains(i) && graph_in[i].contains(j)){
				vals.push_back(1.0/oj2[j]);
				cols.push_back(j);
			}
			else if(oj2[j] == 0){
				vals.push_back(1.0/nodes);
				cols.push_back(j);
			}
		}
		rows.push_back(vals.size());
	}
	graph_in.clear(); // free memory
	/// initialize rank vector
	rank.resize(nodes, 1.0/nodes);
}

const vector<float> &page_rank::getRank() const{
	return rank;
}

int page_rank::getNodes() const{
	return nodes;
}

int page_rank::getEdges() const{
	return edges;
}

float page_rank::compute_page_rank(int n_threads, int iter, float beta){
	cout << "computing page rank" << endl;
	auto time = omp_get_wtime();
	auto c = (1.0 - beta) / nodes;
	for(auto k = 0; k < iter; ++k){
		vector<float> results(nodes, 0.0);
		#pragma omp parallel for if(n_threads > 1) num_threads(n_threads) schedule(dynamic)
		for(auto i = 0; i < nodes; ++i){
			float sum = 0.0;
			#pragma omp parallel for if(n_threads > 1) reduction(+:sum)
			for(auto j = rows[i]; j < rows[i+1]; ++j){
				sum += vals[j] * rank[cols[j]];
			}
			results[i] = beta * sum + c;
		}
		rank = results;
	}
	return omp_get_wtime() - time;
}
