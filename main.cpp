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
#include <memory>
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

// path helper: weights/transformer.h.{layer}.{suffix}
string h_path(int layer, const string& suffix){
    return "weights/transformer.h." + to_string(layer) + "." + suffix;
}

// overall flow is like i have vector of ids, now i need to look up the ids to embeddings, the strore those in a tenor = vector<vector<float>>, then while porcessing 
// x= wte + wpe  for the first transformer head, it is basically take embedding e then do x=wte+wpe  then x*h0= {q,k,v} , we need to do causality that means only 
// look at previous tokens in attention, so we need to update v, q*k of everything  and then we have score of all, then do v=score*(other values)
// do this for every iteration for one block , at the end we need to send to mlp, which is basically  
//using tensor=vector<float>;

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

// a[T][in] * W[in][out] + bias[out]  ->  out[T][out]
// HF Conv1D stores W as [in, out], same as how we dump the txt files
vector<vector<float>> do_matmul(vector<vector<float>>& a, vector<vector<float>>& W, vector<float>& bias){
    int T=a.size();
    int in_dim=W.size();
    int out_dim=W[0].size();
    vector<vector<float>> out(T, vector<float>(out_dim, 0));
    for(int t=0;t<T;t++){
        for(int j=0;j<out_dim;j++){
            float sum=bias[j];
            for(int k=0;k<in_dim;k++){
                sum+=a[t][k]*W[k][j];
            }
            out[t][j]=sum;
        }
    }
    return out;
}

void LayerNorm(vector<vector<float>>& embeddings, const string& weight_path, const string& bias_path){
    //we need to do row-wise
    for(int i=0;i<embeddings.size();i++){
        float mean=0;
        for(int j=0;j<embeddings[0].size();j++){
            mean+=embeddings[i][j];
        }
        mean=mean/768;
        //now do variance
        float variance=0;
        for(int j=0;j<embeddings[0].size();j++){
            variance+=(embeddings[i][j]-mean)*(embeddings[i][j]-mean);
        }
        variance=variance/768;
        // then normalize x̂ᵢ = (xᵢ - mean) / sqrt(variance + ε)
        float eps=1e-5;
        for(int j=0;j<embeddings[0].size();j++){
            embeddings[i][j]=  (embeddings[i][j] - mean)/sqrt(variance+eps);
        }
    }
    // i think this does the normalisation, then we need to do the learned weights : n*ln_w + ln_b
    ifstream file1(weight_path);
    
    vector<float> ln_weights(768); // for ln weights = 768 floats
    vector<float> ln_bias(768);
    float temp; 
    for(int i=0;i<768;i++){
        file1>>temp;
        ln_weights[i]=temp;
    }

    ifstream file2(bias_path);

    for(int i=0;i<768;i++){
        file2>>temp;
        ln_bias[i]=temp;
    }
    // so i have lnweights and lnbias
    // embeddings[i][j]=embeddings[i][j]*lnw[j]+lnb[j];
    for(int i=0;i<embeddings.size();i++){
        for(int j=0;j<embeddings[0].size();j++){
            embeddings[i][j]=(embeddings[i][j]*ln_weights[j]) + ln_bias[j];
        }
    }
}

// attention block for one layer
// input is already ln_1(x), shape [T x 768]
// c_attn -> [T x 2304] = concat(Q,K,V) each 768
// then 12 heads of 64, causal qk^T /sqrt(64), softmax, *V, concat, c_proj
void attention_block(vector<vector<float>>& x, int layer){
    // load c_attn weight [768][2304] and bias [2304]
    vector<vector<float>> c_attn_w(768, vector<float>(2304, 0));
    vector<float> c_attn_b(2304, 0);
    ifstream f1(h_path(layer, "attn.c_attn.weight.txt"));
    float num;
    for(int i=0;i<768;i++){
        for(int j=0;j<2304;j++){
            f1>>num;
            c_attn_w[i][j]=num;
        }
    }
    f1.close();
    ifstream f2(h_path(layer, "attn.c_attn.bias.txt"));
    for(int i=0;i<2304;i++){
        f2>>num;
        c_attn_b[i]=num;
    }
    f2.close();

    // x * c_attn -> [T x 2304], then split into q,k,v each [T x 768]
    vector<vector<float>> qkv=do_matmul(x, c_attn_w, c_attn_b);
    int T=x.size();
    vector<vector<float>> Q(T, vector<float>(768, 0));
    vector<vector<float>> K(T, vector<float>(768, 0));
    vector<vector<float>> V(T, vector<float>(768, 0));
    for(int t=0;t<T;t++){
        for(int j=0;j<768;j++){
            Q[t][j]=qkv[t][j];
            K[t][j]=qkv[t][768+j];
            V[t][j]=qkv[t][1536+j];
        }
    }

    // multi head: 12 heads, each head_dim=64
    int n_head=12;
    int head_dim=64;
    float scale=sqrt(64.0f);
    vector<vector<float>> attn_out(T, vector<float>(768, 0));

    for(int h=0;h<n_head;h++){
        // scores[i][j] = (Q_i · K_j) / sqrt(64), only j<=i (causal)
        vector<vector<float>> scores(T, vector<float>(T, 0));
        for(int i=0;i<T;i++){
            for(int j=0;j<=i;j++){
                float sum=0;
                for(int d=0;d<head_dim;d++){
                    sum+=Q[i][h*head_dim+d]*K[j][h*head_dim+d];
                }
                scores[i][j]=sum/scale;
            }
            // mask future tokens so softmax ~ 0 there
            for(int j=i+1;j<T;j++){
                scores[i][j]=-1e10f;
            }
            // softmax over keys for this query row
            float maxv=scores[i][0];
            for(int j=1;j<T;j++){
                if(scores[i][j]>maxv) maxv=scores[i][j];
            }
            float sumexp=0;
            for(int j=0;j<T;j++){
                scores[i][j]=exp(scores[i][j]-maxv);
                sumexp+=scores[i][j];
            }
            for(int j=0;j<T;j++){
                scores[i][j]/=sumexp;
            }
            // attn_out for this head = scores * V_head
            for(int d=0;d<head_dim;d++){
                float sum=0;
                for(int j=0;j<T;j++){
                    sum+=scores[i][j]*V[j][h*head_dim+d];
                }
                attn_out[i][h*head_dim+d]=sum;
            }
        }
    }

    // c_proj: [T x 768] -> [T x 768]
    vector<vector<float>> c_proj_w(768, vector<float>(768, 0));
    vector<float> c_proj_b(768, 0);
    ifstream f3(h_path(layer, "attn.c_proj.weight.txt"));
    for(int i=0;i<768;i++){
        for(int j=0;j<768;j++){
            f3>>num;
            c_proj_w[i][j]=num;
        }
    }
    f3.close();
    ifstream f4(h_path(layer, "attn.c_proj.bias.txt"));
    for(int i=0;i<768;i++){
        f4>>num;
        c_proj_b[i]=num;
    }
    f4.close();

    x=do_matmul(attn_out, c_proj_w, c_proj_b);
}

// gpt2 gelu: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
float gelu(float x){
    return 0.5f*x*(1.0f+tanh(sqrt(2.0f/M_PI)*(x+0.044715f*x*x*x)));
}

// mlp for one layer: c_fc 768->3072, gelu, c_proj 3072->768
void mlp_block(vector<vector<float>>& x, int layer){
    vector<vector<float>> c_fc_w(768, vector<float>(3072, 0));
    vector<float> c_fc_b(3072, 0);
    float num;
    ifstream f1(h_path(layer, "mlp.c_fc.weight.txt"));
    for(int i=0;i<768;i++){
        for(int j=0;j<3072;j++){
            f1>>num;
            c_fc_w[i][j]=num;
        }
    }
    f1.close();
    ifstream f2(h_path(layer, "mlp.c_fc.bias.txt"));
    for(int i=0;i<3072;i++){
        f2>>num;
        c_fc_b[i]=num;
    }
    f2.close();

    vector<vector<float>> hidden=do_matmul(x, c_fc_w, c_fc_b);
    for(int i=0;i<hidden.size();i++){
        for(int j=0;j<3072;j++){
            hidden[i][j]=gelu(hidden[i][j]);
        }
    }

    vector<vector<float>> c_proj_w(3072, vector<float>(768, 0));
    vector<float> c_proj_b(768, 0);
    ifstream f3(h_path(layer, "mlp.c_proj.weight.txt"));
    for(int i=0;i<3072;i++){
        for(int j=0;j<768;j++){
            f3>>num;
            c_proj_w[i][j]=num;
        }
    }
    f3.close();
    ifstream f4(h_path(layer, "mlp.c_proj.bias.txt"));
    for(int i=0;i<768;i++){
        f4>>num;
        c_proj_b[i]=num;
    }
    f4.close();

    x=do_matmul(hidden, c_proj_w, c_proj_b);
}

// one full transformer block (pre-ln):
// residual=x; x=ln1(x); x=attn(x); x+=residual;
// residual=x; x=ln2(x); x=mlp(x);  x+=residual;
void transformer_block(vector<vector<float>>& x, int layer){
    vector<vector<float>> residual=x;
    LayerNorm(x, h_path(layer, "ln_1.weight.txt"), h_path(layer, "ln_1.bias.txt"));
    attention_block(x, layer);
    do_matadd(x, residual); // residual after attn

    residual=x;
    LayerNorm(x, h_path(layer, "ln_2.weight.txt"), h_path(layer, "ln_2.bias.txt"));
    mlp_block(x, layer);
    do_matadd(x, residual); // residual after mlp
}

// load wte once, keep it for embed + final logits (tied weights)
vector<vector<float>> load_wte(){
    vector<vector<float>> wte(50257, vector<float>(768, 0));
    ifstream file1("weights/transformer.wte.weight.txt");
    if(!file1){
        cout<<"error loading wte\n";
        return wte;
    }
    float num;
    for(int i=0;i<50257;i++){
        for(int j=0;j<768;j++){
            file1>>num;
            wte[i][j]=num;
        }
    }
    file1.close();
    return wte;
}

vector<vector<float>> load_wpe(){
    vector<vector<float>> wpe(1024, vector<float>(768, 0));
    ifstream file2("weights/transformer.wpe.weight.txt");
    if(!file2){
        cout<<"error loading wpe\n";
        return wpe;
    }
    float num;
    for(int i=0;i<1024;i++){
        for(int j=0;j<768;j++){
            file2>>num;
            wpe[i][j]=num;
        }
    }
    file2.close();
    return wpe;
}

vector<vector<float>> do_embed(vector<int32_t>& id, vector<vector<float>>& wte, vector<vector<float>>& wpe){
    // so basically we need to return wte + wpe, tokens less than 1024 for now
    vector<vector<float>> embedding;
    for(int i=0;i<id.size();i++){
        embedding.push_back(wte[id[i]]);
    }
    // add position embeddings for positions 0..T-1
    for(int i=0;i<embedding.size();i++){
        do_rowadd(embedding[i], wpe[i]);
    }
    return embedding;
}

// full forward: embed already done -> 12 blocks -> ln_f
void forward(vector<vector<float>>& x){
    for(int layer=0;layer<12;layer++){
        cout<<"running block h."<<layer<<" ..."<<endl;
        transformer_block(x, layer);
    }
    LayerNorm(x, "weights/transformer.ln_f.weight.txt", "weights/transformer.ln_f.bias.txt");
}

// logits[v] = dot(last_hidden, wte[v])  — tied lm head, no separate matrix
vector<float> lm_head(vector<float>& h, vector<vector<float>>& wte){
    vector<float> logits(50257, 0);
    for(int v=0;v<50257;v++){
        float sum=0;
        for(int d=0;d<768;d++){
            sum+=h[d]*wte[v][d];
        }
        logits[v]=sum;
    }
    return logits;
}

int argmax(vector<float>& logits){
    int best=0;
    for(int i=1;i<logits.size();i++){
        if(logits[i]>logits[best]) best=i;
    }
    return best;
}

unique_ptr<tokenizers::Tokenizer> make_tokenizer(){
    string json=load_file("weights/tokenizer/tokenizer.json");
    return tokenizers::Tokenizer::FromBlobJSON(json);
}

int main(){   
    string s="I like the"; //  for now keep it below 1024 tokens because of wpe restrictions

    auto tokenizer=make_tokenizer();
    vector<int32_t> id=tokenizer->Encode(s);

    cout<<"prompt ids:";
    for(int i=0;i<id.size();i++) cout<<" "<<id[i];
    cout<<endl;

    cout<<"loading wte and wpe..."<<endl;
    vector<vector<float>> wte=load_wte();
    vector<vector<float>> wpe=load_wpe();
    cout<<"wte and wpe loaded!"<<endl;

    vector<vector<float>> embeddings=do_embed(id, wte, wpe);

    // 12 transformer blocks + final ln_f
    forward(embeddings);

    // take last token hidden -> vocab logits via wte
    vector<float> logits=lm_head(embeddings.back(), wte);
    int next_id=argmax(logits);
    id.push_back(next_id);

    string next_piece=tokenizer->Decode({next_id});
    string full_text=tokenizer->Decode(id);

    cout<<"next token id: "<<next_id<<" -> \""<<next_piece<<"\""<<endl;
    cout<<"decoded so far: "<<full_text<<endl;
    return 0;
}
