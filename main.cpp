#include <vector>
#include <iostream>
#include "page_rank.h"

using namespace std;

int main() {
	auto *pr = new page_rank("../datasets/p2p-Gnutella25.txt");
	auto rank = pr->compute_page_rank(50, 0.85);
	for(auto i = 0; i < rank.size(); ++i){
		cout << "rank[" << i << "]= " << rank[i] << endl;
	}
	return 0;
}