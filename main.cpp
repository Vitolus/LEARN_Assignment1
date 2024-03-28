#include <vector>
#include <iostream>
#include "page_rank.h"

using namespace std;

void writeCSV(const string &filename, const vector<float> &times, const vector<float> &speedups){
	ofstream file(filename);
	if(!file.is_open()){
		cout << "file not found" << endl;
		return;
	}
	file << "n threads,time,speedup" << endl;
	for(auto i = 1; i <= times.size(); ++i){
		file << i << "," << times[i-1] << "," << speedups[i-1] << endl;
	}
	file.close();
}

int main(int argc, char *argv[]) { // filepath, n_threads
	if(argc != 3){
		cout << "Usage: " << argv[0] << " <filepath> <n_threads>" << endl;
		return 1;
	}
	vector<float> times(stoi(argv[2]), 0.0);
	vector<float> speedups(stoi(argv[2]), 0.0);
	auto *pr = new page_rank(argv[1]);
	auto nodes = pr->getNodes();
	auto edges = pr->getEdges();
	vector<float> *rank;
	for(auto i = 1; i <= stoi(argv[2]); ++i){
		auto time = omp_get_wtime();
		rank = pr->compute_page_rank(i, 50, 0.85);
		times[i-1] = omp_get_wtime() - time;
		speedups[i-1] = times[0] / times[i-1];
	}
	for(auto i = 0; i < 5; ++i){
		cout << "rank[" << i << "]= " << (*rank)[i] << endl;
	}
	string csvfile = argv[1];
	auto pos = csvfile.find_last_of('.');
	csvfile = (pos != string::npos) ? csvfile.substr(0, pos) : csvfile;
	csvfile += "-speedup.csv";
	writeCSV(csvfile, times, speedups);
	cout << "number of nodes: " << nodes << endl;
	cout << "number of edges: " << edges << endl;
	cout << "sparsity rate: " << static_cast<float>(edges)/(nodes*(nodes-1)) << endl;
	for(auto i = 1; i <= stoi(argv[2]); ++i){
		cout << "n_threads= " << i << " time= " << times[i-1] << " speedup= " << speedups[i-1] << endl;
	}
	return 0;
}