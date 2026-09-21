// #include<bits/stdc++.h>

// using namespace std;

// vector<float> load_weights(const string& filename){
//     ifstream file(filename);
//     if (!file) {
//         throw std::runtime_error("Could not open " + filename);
//     }
//     vector<float> weights;
//     float value;
//     while (file >> value) {
//         weights.push_back(value);
//     }
//     return weights;
// }

// void load_model(const string& filename){
//     vector<float> weights = load_weights(filename);
//     cout << "Loaded " << weights.size() << " weights" << endl;
// }

// int main(){
// 	// let me try to load the model
// 	vector<float> wte=load_weights("weights/transformer.wte.weight.txt");
//     vector<float> wpe=load_weights("weights/transformer.wpe.weight.txt");
//     cout << "Loaded " << wte.size() << " weights" << endl;
//     cout << "Loaded " << wpe.size() << " weights" << endl;


	
// 	return 0;
// }




#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "tokenizers_cpp.h"


using namespace std;

std::string load_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Could not open: " + path);
    }

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

// overall flow is like i have vector of ids, now i need to look up the ids to embeddings, the strore those in a tenor = vector<vector<float>>, then while porcessing 
// x= wte + wpe  for the first transformer head, it is basically take embedding e then do x=wte+wpe  then x*h0= {q,k,v} , we need to do causality that means only 
// look at previous tokens in attention, so we need to update v, q*k of everything  and then we have score of all, then do v=score*(other values)
// do this for every iteration for one block , at the end we need to send to mlp, which is basically  
//using tensor=vector<float>;


vector<vector<float>> do_embed(vector<int32_t>& id){
    //logic to return embedding
    vector<vector<float>> arr;
    return arr;
}



vector<int32_t> init_tokenizer(string s){
    vector<int32_t> ids;
    try {
        string json =
            load_file("weights/tokenizer/tokenizer.json");

        auto tokenizer =
            tokenizers::Tokenizer::FromBlobJSON(json);

        string text = s;
        ids=tokenizer->Encode(text);
            return ids;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return ids;
}

int main(){   
    string s="I like the cat is veryhappy";
    vector<int32_t> id=init_tokenizer(s);// i give string and get vector of ids
    vector<vector<float>> embeddings=do_embed(id);
    

    return 0;
}