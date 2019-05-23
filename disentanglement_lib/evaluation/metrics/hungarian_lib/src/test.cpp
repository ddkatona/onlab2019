#include <iostream>
#include <fstream>

#include "hungarian.h"
#include "problems.h"
#include "mincost_maxflow.tpp"
#include "geom_object.h"

/**
 * Compile with: g++ -O3 -std=c++17 -o bin/test src/test.cpp -lstdc++fs
 *           or: g++ -g -std=c++17 -o bin/test src/test.cpp -lstdc++fs
 * 
 */

using namespace std;

void test1(){
    Hungarian_method method = Hungarian_method();
    method.read_mtx("data/test/test1.in",100);
    method.init();

    method.sparse_all(2);
    auto p=method.Gamma();
    cout<<p.first<<" "<<p.second<<" "<<((double) p.first/p.second)*100.0<<endl;
}
void test2(){
    pair<int,int> pair0={28,28};
    string folder1="data/mnist/train/data";
    string folder2="models/wgan/generator_1000/data";

    vector<vector<double> > v1(28,vector<double>(28,0));
    vector<vector<double> > v2(28,vector<double>(28,0));

    v1[1][1]=1;
    v2[2][2]=1;

    cout<<euclidian_dist(v1,v2)<<endl;


    generate_graph(100,pair0, folder1, folder2);

    const char* filename="/tmp/mnist_mtx.txt";
    Hungarian_method method = Hungarian_method();
    method.run(filename,"data/test/test3.out",2000);

    vector<vector<double> > mtx;
    beolv<double>(mtx,filename,2000);
    Flow<double> f;
    auto p=f.minCost_maxMatching_flow(mtx);
    cout<<p.first<<" "<<p.second<<endl;
    std::cout<<"===== END ====="<<endl;
}
int main(){
    ofstream myfile("./valami.txt");
    myfile<<"100\n"<<"Szoveg\n";
    cout<<"100\n"<<"Szoveg\n";
    myfile.close();
    return 0;
}