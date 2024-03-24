#include <vector>
#include <iostream>
#include "page_rank.h"

using namespace std;

int main(int argc, char *argv[]) { // filepath, n_threads
	if(argc != 3){
		cout << "Usage: " << argv[0] << " <filepath> <n_threads>" << endl;
		return 1;
	}
	auto *pr = new page_rank(argv[1]);
	auto time = omp_get_wtime();
	pr->compute_page_rank(stoi(argv[2]), 50, 0.85);
	time = omp_get_wtime() - time;
	auto rank = pr->getRank();
	for(auto i = 0; i < 5; ++i){
		cout << "rank[" << i << "]= " << rank[i] << endl;
	}
	for(auto i = rank.size() - 5; i < rank.size(); ++i){
		cout << "rank[" << i << "]= " << rank[i] << endl;
	}
	cout << "time to perform page rank: " << time << endl;
	return 0;
}