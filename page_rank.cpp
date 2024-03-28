#include "page_rank.h"

vector<ulong> *page_rank::outDegree(const unordered_map<ulong, unordered_set<ulong>> *graph_out) const{
	auto *oj = new vector<ulong>(nodes, 0);
	for(const auto& i : *graph_out){
		(*oj)[i.first] = i.second.size();
	}
	return oj;
}

page_rank::page_rank(const string &filename) : filename(filename), nodes(0), edges(0),
rows(new vector<ulong>()), cols(new vector<ulong>()), vals(new vector<float>()){
	/// first pass to get the dimension of the graph
	cout << "first pass reading file" << endl;
	ifstream file(this->filename);
	if(!file.is_open()){
		cout << "file not found" << endl;
		return;
	}
	string line;
	while(getline(file, line)){
		istringstream iss(line);
		long start, arrive;
		iss >> start >> arrive;
		nodes = max(nodes, max(start, arrive) + 1);
		++edges;
	}
	file.close();
	auto *graph_out = new unordered_map<ulong, unordered_set<ulong>>();
	auto *graph_in = new unordered_map<ulong, unordered_set<ulong>>();
	/// second pass to initialize graph with 1 where there is an edge
	cout << "second pass reading file" << endl;
	file.open(this->filename);
	if(!file.is_open()){
		cout << "file not found" << endl;
		return;
	}
	while(getline(file, line)){
		istringstream iss(line);
		long start = 0, arrive = 0;
		iss >> start >> arrive;
		(*graph_out)[start].insert(arrive);
		(*graph_in)[arrive].insert(start);
	}
	file.close();
	/// compute out degree vector
	cout << "computing out degree vector" << endl;
	vector<ulong> *oj = outDegree(graph_out);
	delete graph_out;
	graph_out = nullptr;
	/// create CSR transition matrix
	cout << "creating CSR transition matrix" << endl;
	rows->push_back(0);
	for(auto i = 0; i < nodes; ++i){
		auto it_i = graph_in->find(i);
		for(auto j = 0; j < nodes; ++j){
			if(it_i != graph_in->end() && it_i->second.contains(j)){
				vals->push_back(1.0/((*oj)[j]));
				cols->push_back(j);
			}
			else if((*oj)[j] == 0){
				vals->push_back(1.0/nodes);
				cols->push_back(j);
			}
		}
		rows->push_back(vals->size());
	}
	rank = new vector<float>(nodes, 1.0/nodes);
}

long page_rank::getNodes() const{
	return nodes;
}

long page_rank::getEdges() const{
	return edges;
}

vector<float> *page_rank::compute_page_rank(int n_threads, int iter, float beta){
	cout << "computing page rank" << endl;
	auto c = (1.0 - beta) / nodes;
	auto *results = new vector<float>(nodes, 0.0);
	for(auto k = 0; k < iter; ++k){
		fill(results->begin(), results->end(), 0.0); // Reset values to 0
		#pragma omp parallel for if(n_threads > 1) num_threads(n_threads) schedule(dynamic)
		for(auto i = 0; i < nodes; ++i){
			float sum = 0.0;
			#pragma omp parallel for if(n_threads > 1) reduction(+:sum)
			for(auto j = (*rows)[i]; j < (*rows)[i+1]; ++j){
				sum += (*vals)[j] * (*rank)[(*cols)[j]];
			}
			(*results)[i] = beta * sum + c;
		}
		swap(*rank, *results);
	}
	return rank;
}