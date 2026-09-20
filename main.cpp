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

int main()
{
    try {
        std::string json =
            load_file("weights/tokenizer/tokenizer.json");

        auto tokenizer =
            tokenizers::Tokenizer::FromBlobJSON(json);

        std::string text = "I like the cat";

        std::vector<int32_t> ids =
            tokenizer->Encode(text);

        std::cout << "Input: " << text << "\n";

        std::cout << "Token IDs: ";

        for (int32_t id : ids) {
            std::cout << id << " ";
        }

        std::cout << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}