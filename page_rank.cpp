#include "page_rank.h"

vector<int> page_rank::outDegree(const vector<short>& graph) const{
	vector<int> oj(dim, 0);
	for (auto j = 0; j < dim; ++j) {
		for (auto i = 0; i < dim; ++i){
			oj[j] += graph[i*dim + j];
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
//TODO: consider unordered_map of vectors insted of vector of vectors to store only non zero values
	vector<short> graph(dim*dim, 0);
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
		graph[arrive*dim +start] = 1;
		break;
	}
	#pragma omp parallel num_threads(omp_get_max_threads()) default(none) shared(file, line, iss, graph)
	{
		#pragma omp single
		while(getline(file, line)){
			#pragma omp task default(none) shared(line, graph) private(iss)
			{
				iss.str(line);
				int start, arrive;
				iss >> start >> arrive;
//TODO: gives segmentation fault if arrive*dim + start is not in the graph
				graph[arrive*dim +start] = 1;
			}
		}
	}
	file.close();
	/// compute out degree vector
	cout << "computing out degree vector" << endl;
	vector<int> oj = outDegree(graph);
	/// create CSR transition matrix
	rows = {0};
	for(auto i = 0; i < dim; ++i){
		for(auto j = 0; j < dim; ++j){
			if(graph[i*dim +j] == 1){
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
	cout << "computing page rank" << endl;
	auto c = (1.0 - beta) / dim;
	for(auto k = 0; k < iter; ++k){
		vector<float> results(dim, 0.0);
		#pragma omp parallel for if(n_threads > 1) num_threads(n_threads) schedule(dynamic) \
		default(none) shared(beta, c, results, n_threads)
		for(auto i = 0; i < dim; ++i){
			float sum = 0.0;
			#pragma omp parallel for if(n_threads > 1) reduction(+:sum) default(none) shared(i)
			for(auto j = rows[i]; j < rows[i+1]; ++j){
				sum += vals[j] * rank[cols[j]];
			}
			results[i] = static_cast<float>(beta * sum + c);
		}
		rank = results;
	}
}
