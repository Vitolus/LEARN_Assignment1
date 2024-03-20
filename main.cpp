#include <vector>
#include <iostream>
#include "page_rank.h"

// Function to calculate outdegree of each node
vector<int> outDegree(const vector<vector<int>>& graph) {
	vector<int> outdegree(graph.size(), 0);
	for (int j = 0; j < graph.size(); ++j) {
		for (const auto & i : graph) {
			outdegree[j] += i[j];
		}
	}
	return outdegree;
}

// Function to create CSR representation of M
void createCSR(const vector<vector<int>>& graph) {
	vector<int> outdegree = outDegree(graph);
	int n = graph.size();

	vector<double> values;
	vector<int> column_indices;
	vector<int> row_pointers = {0};

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (graph[i][j] == 1) {
				values.push_back(1.0 / outdegree[j]);
				column_indices.push_back(j);
			}else if(outdegree[j] == 0){
				values.push_back(1.0 / n);
				column_indices.push_back(j);
			}
		}
		row_pointers.push_back(values.size());
	}

	// Print CSR representation
	cout << "Values: ";
	for (double v : values) cout << v << " ";
	cout << "\nColumn Indices: ";
	for (int idx : column_indices) cout << idx << " ";
	cout << "\nRow Pointers: ";
	for (int ptr : row_pointers) cout << ptr << " ";
	cout << "\n";
}

int main() {
	// Example graph
	vector<vector<int>> graph = {
			{0, 0, 0, 1},
			{0, 0, 1, 0},
			{1, 0, 0, 0},
			{0, 0, 1, 0}
	};
	
	vector<vector<float>> M = {
			{0, 0.25, 0, 1},
			{0, 0.25, 0.5, 0},
			{1, 0.25, 0, 0},
			{0, 0.25, 0.5, 0}
	};

	createCSR(graph);
/*
	auto *pr = new page_rank("../datasets/p2p-Gnutella25.txt");
	auto rank = pr->compute_page_rank(50, 0.85);
	for(auto &i : rank){
		cout << i << endl;
	}
*/
	return 0;
}