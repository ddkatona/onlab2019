#pragma once 
#include <iostream>
#include <vector>
#include <fstream>
#include <assert.h>

//#include "glpk.h"

using namespace std;

void knapSack(int W, vector<int>& wt, vector<int>& sizes);

template<typename Type>
void create_min_cost_perf_match_problem(vector<vector<Type> >& mtx);

template<typename Type>
class Flow{
    bool dijkstra( int n, int s, int t );
public:
    // Miinimum Cost Maximum Flow
    int mcmf3( int n, int s, int t, int &fcost );
    int run_mcmf(string filename);
    pair<int,int> minCost_maxMatching_flow(vector<vector<Type> > mtx);

};






// =============== HELPER FUNCTIONS =================
template<typename Type>
// ===== Read matrix from a given file =====
void beolv(vector<vector<Type> >& v,string filename,int maxN){
    ifstream myfile(filename.c_str());
    int N,M;
    myfile>>N>>M;
    v.resize(min(N,maxN),vector<Type>(min(M,maxN)));

    for(int i=0;i<N;i++){
        int x;
        myfile>>x;
        for(int j=0;j<M;j++){
            Type temp;
            myfile>>temp;
            if(i<maxN && j<maxN) v[i][j]=temp;
        }
    }
}

