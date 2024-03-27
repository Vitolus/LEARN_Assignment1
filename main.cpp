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
	cout << "dimension of graph: " << nodes << endl;
	cout << "dimension of graph in ram: " << static_cast<float>(sizeof(short)*nodes*nodes)/(1024*1024*1024) << " GB" << endl;
	cout << "dimension of graph with only non zero values in ram: " << static_cast<float>(sizeof(short)*edges*2)/(1024*1024) << " MB" << endl;
	cout << "sparisity rate: " << static_cast<float>(edges)/(nodes*(nodes-1)) << endl;
	for(auto i = 1; i <= stoi(argv[2]); ++i){
		auto time = pr->compute_page_rank(i, 50, 0.85);
		times[i-1] = time;
		speedups[i-1] = times[0] / times[i-1];
	}
	auto rank = pr->getRank();
	for(auto i = 0; i < 5; ++i){
		cout << "rank[" << i << "]= " << rank[i] << endl;
	}
	for(auto i = rank.size() - 5; i < rank.size(); ++i){
		cout << "rank[" << i << "]= " << rank[i] << endl;
	}
	string csvfile = argv[1] + (string)"-perf.csv";
	writeCSV(csvfile, times, speedups);
	cout << "time to perform page rank:" << endl;
	for(auto i = 1; i <= stoi(argv[2]); ++i){
		cout << "n_threads= " << i << " time= " << times[i-1] << " speedup= " << speedups[i-1] << endl;
	}
	return 0;
}