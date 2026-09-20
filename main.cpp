#include<bits/stdc++.h>

using namespace std;

vector<float> load_weights(const string& filename){
    ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open " + filename);
    }
    vector<float> weights;
    float value;
    while (file >> value) {
        weights.push_back(value);
    }
    return weights;
}

void load_model(const string& filename){
    vector<float> weights = load_weights(filename);
    cout << "Loaded " << weights.size() << " weights" << endl;
}

int main(){
	// let me try to load the model
	vector<float> wte=load_weights("weights/transformer.wte.weight.txt");
    vector<float> wpe=load_weights("weights/transformer.wpe.weight.txt");
    cout << "Loaded " << wte.size() << " weights" << endl;
    cout << "Loaded " << wpe.size() << " weights" << endl;


	
	return 0;
}
