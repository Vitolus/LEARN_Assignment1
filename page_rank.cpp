#include "page_rank.h"

float page_rank::out_degree(vector<vector<float>> &graph, int col){
	float degree = 0.0;
	for(const auto &row : graph){
		degree += row[col];
	}
	return degree;
}

page_rank::page_rank(const string &filename) : filename(filename){
	/// header processing
	ifstream file(this->filename);
	if(!file.is_open()){
		cout << "file not found" << endl;
		return;
	}
	string line;
	for(short i = 0; i < 2; ++i){
		getline(file, line); // skip first two header lines
	}
	getline(file, line); // line with number of nodes and edges
	istringstream iss(line);
	string n_nodes;
	while(iss >> n_nodes){
		if(n_nodes == "Nodes:"){
			iss >> n_nodes;  // get number of nodes
			break;
		}
	}
	getline(file, line); // skip last header line
	this->dim = stoi(n_nodes);
	vector<vector<short>> graph(this->dim, vector<short>(this->dim, 0.0));
	/// initialize graph with 1 where there is an edge
	while(getline(file, line)){
		iss.str(line);
		int start, arrive;
		iss >> start >> arrive;
		graph[arrive][start] = 1;
	}
	file.close();

//TODO: take times
	/// csr approach
	rows.push_back(0);  // first element is always 0
	for(auto j =0; j < this->dim; ++j){  // iterate over columns
		float oj = 0.0;  
		for(auto i = 0; i < this->dim; ++i){  // iterate over rows
			if(graph[i][j] == 1){  // if there is an edge
				cols.push_back(i);  // add row index to cols
				++oj;  // increment out degree
			}
		}
		float trans = (oj > 0) ? 1.0/oj : 1.0/this->dim;  // if out degree is 0, set transition to 1/n else 1/out degree
		vals.insert(vals.end(), oj, trans);  // insert oj times transition value
		rows.push_back(cols.size());  // add number of elements in cols to rows
	}

	/// matrix approach
	/*
	cout << "start init" << endl;
	this->matrix.resize(this->dim, vector<float>(this->dim, 0.0));

	cout << "start init oj" << endl;
	vector<float> oj(this->dim, 0.0);
	for(int i =0; i< this->dim; ++i){
		oj[i] = out_degree(graph, i);
	}

	cout << "start init csr" << endl;
	for(auto i = 0; i < this->dim; ++i){
		for(auto j = 0; j < this->dim; ++j){
			if(graph[i][j] == 1){
				float val = 1.0/oj[j];
				this->matrix[i][j] = val;
			}
			else if(oj[j] == 0){
				float val = 1.0/this->dim;
				this->matrix[i][j] = val;
			}
		}
	}
	*/
	this->rank.resize(this->dim, 1.0/this->dim);
}

vector<float> page_rank::compute_page_rank(int iter, float beta){
	/*
	int n = M.row_ptr.size() - 1;
    std::vector<double> v(n, 1.0 / n);

    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<double> v_new(n, (1 - b) / n);
        for (int j = 0; j < n; ++j) {
            for (int idx = M.row_ptr[j]; idx < M.row_ptr[j + 1]; ++idx) {
                int i = M.col_idx[idx];
                v_new[i] += b * M.values[idx] * v[j];
            }
        }

        double err = 0;
        for (int i = 0; i < n; ++i) {
            err += std::abs(v[i] - v_new[i]);
        }
        if (err < tol) {
            return v_new;
        }

        v = v_new;
    }

    return v;
	 */
	float c = (1.0 - beta) / this->dim;
	for(auto k = 0; k < iter; ++k){
		vector<float> results(this->dim, 0.0);
		cout << "iter: " << k << endl;

		/// matrix approach
		for(auto i = 0; i < this->dim; ++i){
			float sum = 0.0;
			for(auto j = 0; j < this->dim; ++j){
				sum += this->matrix[i][j] * this->rank[j];
			}
			results[i] = beta * sum + c;
		}

		/// csr approach
//TODO: csr approach
		this->rank = results;
	}
	return this->rank;
}
