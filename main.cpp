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
#include <cmath>

#include "tokenizers_cpp.h"


using namespace std;

string load_file(const string& path)
{
    ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Could not open: " + path);
    }

    return string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

// overall flow is like i have vector of ids, now i need to look up the ids to embeddings, the strore those in a tenor = vector<vector<float>>, then while porcessing 
// x= wte + wpe  for the first transformer head, it is basically take embedding e then do x=wte+wpe  then x*h0= {q,k,v} , we need to do causality that means only 
// look at previous tokens in attention, so we need to update v, q*k of everything  and then we have score of all, then do v=score*(other values)
// do this for every iteration for one block , at the end we need to send to mlp, which is basically  
//using tensor=vector<float>;


void LayerNorm(vector<vector<float>>& embeddings){// need mean and variance and then load learned parameters
    //we need to do row-wise
    float mean=0;
    for(int i=0;i<embeddings.size();i++){
        for(int j=0;j<embeddings[0].size();j++){
            mean+=embeddings[i][j];
        }
        mean=mean/768;
        //now do variance
        float variance=0;
        for(int j=0;j<embeddings.size();j++){
            variance+=(embeddings[i][j]-mean)*(embeddings[i][j]-mean);
        }
        variance=variance/768;
        // then normalize x̂ᵢ = (xᵢ - mean) / sqrt(variance + ε)
        float eps=1e-5;
        for(int j=0;j<embeddings[0].size();j++){
            embeddings[i][j]=  (embeddings[i][j] - mean)/sqrt(variance+eps);
        }
    }
    // i think this does the normalisation, then we need to do the learned weights : n*ln1_w + ln_1b
}


void do_rowadd(vector<float> &a, vector<float>& b){ // add rows of matrix
    for(int i=0;i<a.size();i++){
        a[i]+=b[i];
    }
}

void do_matadd(vector<vector<float>>& embedding,vector<vector<float>>& wpe){
    for(int i=0;i<embedding.size();i++){
        do_rowadd(embedding[i],wpe[i]);
    }
}




vector<vector<float>> do_embed(vector<int32_t>& id){
    //logic to return embedding
    // need to do both wte and wpe
    vector<vector<float>> wte(50257,vector<float>(768,0));
    vector<vector<float>> wpe(1024,vector<float>(768,0));
    
    
    //loading wte
    ifstream file1("weights/transformer.wte.weight.txt");
    if(!file1){
        cout<<"error in file1\n";
        return wte;  // debug this
    }
    float num;
    // for wte
    for(int i=0;i<50257;i++){ 
        for(int j=0;j<768;j++){
            file1>>num;
            wte[i][j]=num;
        }
    }
    file1.close();
    ifstream file2("weights/transformer.wpe.weight.txt");
    if(!file2){
        cout<<"error in file 2\n";
        return wte;  // debug this
    }
    
    //for wpe
    for(int i=0;i<1024;i++){ 
        for(int j=0;j<768;j++){
            file2>>num;
            wpe[i][j]=num;
        }
    }
    file2.close();
    cout<<" wte and wpe loaded!"<<endl;
    // so basically we need to return wte + wpe, tokens less than 1024 for now
    vector<vector<float>> embedding;
    for(int i=0;i<id.size();i++){
        embedding.push_back( wte[id[i]] );
    }
    do_matadd(embedding,wpe);
    return embedding;
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
    string s="I like the cat is very happy"; //  for now keep it below 1024 tokens because of wpe restrictions


    vector<int32_t> id=init_tokenizer(s);// i give string and get vector of ids
    vector<vector<float>> embeddings=do_embed(id); // i give vector of ids and get return of embeddings(x) with positional addition also
    LayerNorm(embeddings);
    // i think this does the normalisation, then we need to do the learned weights : n*ln1_w + ln_1b
    cout<<"done till latest"<<endl;
    return 0;
}