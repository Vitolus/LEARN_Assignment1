#include <vector>
#include <iostream>
#include "page_rank.h"

using namespace std;

int main(int argc, char *argv[]) { // filepath, n_threads
	if(argc != 3){
		cout << "Usage: " << argv[0] << " <filepath> <n_threads>" << endl;
		return 1;
	}
	vector<float> times(stoi(argv[2], 0));
	vector<float> speedups(stoi(argv[2], 0));
	auto *pr = new page_rank(argv[1]);
	for(auto i = 1; i <= stoi(argv[2]); ++i){
		auto time = omp_get_wtime();
		pr->compute_page_rank(i, 50, 0.85);
		times[i-1] = omp_get_wtime() - time;
		speedups[i-1] = times[0] / times[i-1];
	}
	auto rank = pr->getRank();
	for(auto i = 0; i < 5; ++i){
		cout << "rank[" << i << "]= " << rank[i] << endl;
	}
	for(auto i = rank.size() - 5; i < rank.size(); ++i){
		cout << "rank[" << i << "]= " << rank[i] << endl;
	}
	cout << "time to perform page rank:" << endl;
	for(auto i = 1; i <= stoi(argv[2]); ++i){
		cout << "n_threads= " << i << " time= " << times[i-1] << " speedup= " << speedups[i-1] << endl;
	}
	return 0;
}