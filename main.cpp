#include <vector>
#include <iostream>
#include "page_rank.h"

using namespace std;

int main(int argc, char *argv[]) { // filepath, n_threads
	if(argc != 3){
		cout << "Usage: " << argv[0] << " <filepath> <n_threads>" << endl;
		return 1;
	}
	auto *pr = new page_rank(argv[1], stoi(argv[2]));
	auto rank = pr->compute_page_rank(50, 0.85);
	for(auto i = 0; i < rank.size(); ++i){
		cout << "rank[" << i << "]= " << rank[i] << endl;
	}
	return 0;
}