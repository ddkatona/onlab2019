// ==================================================================
//                       HUNGARIAN METHOD
// ==================================================================
// Description: 
//    --> Calculates the minimum cost, perfect matching (MCPM) 
//        with the Hungerian method (see Kuhn 1955).
// Input:
//    --> The weighted adjacency matrix of a bipartite graph:
//        mtx[i][j]=c_ij is the weight, that we choose (i,j) edge.
//        (If we want to solve an assigment problem, this is the 
//         cost that worker i does the job j)
// Output:
//    --> The minimum cost perfect matching as vector<pair<int,int> >
// Author:
//    --> Domonkos Czifra, ELTE (Budapest, Hungary), 2018
//




#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <limits>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <limits.h>

using namespace std;


class Hungarian_method{
    enum{S,T};

public:
    Hungarian_method(){}

    void read_mtx(string filename, int maxN){
        fstream myfile(filename.c_str());

        myfile>>N>>M;

        maxN=min(N,maxN);

        mtx.resize(maxN,vector<long double>(maxN,0.0));
        for(int i=0;i<N;i++){
            int trash;
            myfile>>trash;
            for(int j=0;j<M;j++){
                long double temp;
                myfile>>temp;
                //int t=(int) temp;
                long double t = temp;
                if(i<maxN && j<maxN){
                    mtx[i][j]=(long double) t;
                }
            }
        }
        N=maxN;
        M=maxN;

        //cout<<"end"<<endl;
        myfile.close();
    }
    vector<vector<long double> > get_mtx(){
        return mtx;
    }
    void init(){
        // ===== Initialization =====
        y[S].resize(N,0.0);
        y[T].resize(M,0.0);
        //tight.resize(N,vector<int>(M,-1));
        uncovered[S].resize(N,true);
        uncovered[T].resize(M,true);
        matching[S].resize(N,-1);
        matching[T].resize(M,-1);
    }

    double run(std::string file_in, std::string file_out, int N){
        read_mtx(file_in, N);
        init();
        alternating_path();
        double ret=print_matching(file_out);
        return ret;
    }

    template<typename T>
    class Compare{
        bool operator()(T& elem1, T& elem2){
            return elem1<elem2;
        }
    };
    /*
     * Description:
     *     --> Sort the neighbours by weight, keep the N most relevant
     */
    void sparse_neighbours(int node_ID,int N_){
        vector<long double> temp;
        for(int i=0;i<mtx[node_ID].size();i++){
            temp.push_back(mtx[node_ID][i]);
        }
        nth_element(temp.begin(),temp.begin()+N_,temp.end());
        for(int i=0;i<mtx[node_ID].size();i++){
            if(mtx[node_ID][i]>temp[N_-1]){
                mtx[node_ID][i]=INT_MAX;
            }
        }
    }
    void sparse_all(int N){
        for(int i=0;i<mtx.size();i++){
            sparse_neighbours(i,N);
        }
    }
    // Returns the (Deficit,M) pair, where the deficit is the 
    pair<int,int> Gamma(){
        int not_covered_num=0;
        for(int i=0;i<M;i++){
            bool covered=false;
            for(int j=0;j<N;j++){
                if(mtx[j][i]<99999.9){
                    covered=true;
                    break;
                }
            }
            if(not covered) not_covered_num++;
        }
        return {not_covered_num,M};

    }

    long double modify_y(long double delta,vector<bool> (&reachable)[2]){
        for(int i=0;i<reachable[S].size();i++){
            if(reachable[S][i]){
                y[S][i]+=delta;
            }
        }

        for(int i=0;i<reachable[T].size();i++){
            if(reachable[T][i]){
                y[T][i]-=delta;
            }
        }
    }
    
    long double find_delta(vector<bool> (&reachable)[2]){
        long double min = -1;
        // ===== For every element of R_S
        for(int i=0;i<N;i++){
            if(reachable[S][i]){
                // ===== For every element of R_T
                for(int j=0;j<M;j++){
                    if(!reachable[T][j]){
                        long double new_min=mtx[i][j]-y[S][i]-y[T][j];
                        if(min==-1 || new_min<min){
                            min=new_min;
                        }
                    }
                }
            }
        }
        return min;
    }

    long double y_sum(){
        long double sum=0;
        for(auto s:y[S]) sum+=s;
        for(auto t:y[T]) sum+=t;
        return sum;
    }

    double print_matching(std::string filename, bool std_out=false){
        ofstream myfile(filename.c_str());
        long double sum=0;
        for(int i=0;i<N;i++){
            if(matching[S][i]>-1){
                int v=matching[S][i];
                if(std_out) std::cout<<i<<" "<<v<<" "<<mtx[i][v]<<endl;
                myfile<<i<<" "<<v<<" "<<mtx[i][v]<<endl;
                sum+=mtx[i][v];
            }
        }
        //if(std_out)
        cout<<"The min cost matchig is: "<<sum<<endl;
        myfile.close();
        return sum;
    }

    // ===== Alternating path for M matching =====
    // (between uncovered_S and uncovered_T, using only tight edges)
    vector<pair<int,int> > alternating_path(){
        vector<pair<int,int> > ret;

        // ===== Init reachable[S] with uncovered nodes =====
        vector<bool> reachable[2];
        reachable[S].resize(N,false);
        reachable[T].resize(M,false);

        vector<int> que[2];

        for(int i=0;i<uncovered[S].size();i++){
            if(uncovered[S][i]){
                reachable[S][i]=true;
                que[S].push_back(i);
            }
        }

        // ===== Init paranets =====
        vector<int> parents[2];
        parents[S].resize(N,-1);
        parents[T].resize(M,-1);

        // ===== Find alternating path with BFS 
        vector<int> uncov;
        int stop=-1;
        while(1 && uncov.size()==0){
            // ===== Reachable from S =====
            if(que[S].size()==0) break;
            que[T].clear();
            for(int i=0;i<que[S].size() && uncov.size()==0;i++){
                int s=que[S][i];
                for(int j=0;j<M && uncov.size()==0;j++){
                    long double c_uv=mtx[s][j];
                    long double y_u=y[S][s];
                    long double y_v=y[T][j];
                    if( abs(c_uv - (y_u+y_v))< 0.000001 && parents[T][j]==-1)
                    {
                        reachable[T][j]=true;
                        que[T].push_back(j);
                        parents[T][j]=s;
                        if(uncovered[T][j]){
                            uncov.push_back(j);
                        }
                    }
                }
            }
            if(que[T].size()==0) break;

            // ===== Reachable from T, using only the matching =====
            que[S].clear();
            for(int j=0;j<que[T].size();j++){
                int u=que[T][j];
                int v=matching[T][u];
                if(parents[S][v] == -1 && matching[T][u]>-1){
                    reachable[S][v]=true;
                    que[S].push_back(v);
                    parents[S][v]=u;
                }
                
            }
        }

        // ===== Find alternating path with help of the parents =====
        
        if(uncov.size() == 0){
            long double delta=find_delta(reachable);
            //cout<<"delta "<<delta<<endl;
            if(delta==-1){
                for(int i=0;i<N;i++){
                    if(matching[S][i]>-1){
                        int v=matching[S][i];
                        ret.push_back({i,v});
                    }
                }
                return ret;
            }
            else{
                modify_y(delta,reachable);
                return alternating_path();
            }
            //return ret;
        }
        else{
            int act=uncov[0];
            do{
                int v=parents[T][act];
                uncovered[T][act]=false;
                uncovered[S][v]=false;
                matching[T][act]=v;
                matching[S][v]=act;
                ret.push_back({act,v});
                act=parents[S][v];

            }while(act!=-1);
            return alternating_path();
            //return ret;
        }

    }


private:
    vector<bool> uncovered[2];
    vector<int> matching[2];

    vector<vector<long double> > mtx;
    //vector<vector<int> > tight;
    vector<long double> y[2];
    int N,M;
};

