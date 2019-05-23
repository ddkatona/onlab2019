
// ==================================================================
//                 EXAMPLE USAGE OF THE HUNGARIAN METHOD
// ==================================================================
// Description:
//    --> Shows the usage, and geneartes example data
//        for Hungarian method. Such as:
//        * Ramdom points from circles
//        * Random points from lines
//        * MNIST data
//    --> Contains different GAN evaluation metrics, such as 
//        * Minimum Cost Perfect Matching (Hungarian/Flow mode)
//        * Deficit Score
//        * FlowDeficit Score: dropout some edges, and find a 
//            mincost maxflow. The avarage 
// Dependencies:
//    --> c++17, filesystem
// Complile and run with: 
//    --> g++ -O3 -std=c++17 -o bin/main src/main.cpp -lstdc++fs
//    --> bin/main --ini params/params.ini
// Author:
//    --> Domonkos Czifra, ELTE (Budapest, Hungary), 2019
//

// ==================================================================
//                           TOY DATASETS
// ==================================================================
#include <string>
#include <tuple>
#include <sstream>
#include <experimental/filesystem>

#include "hungarian.h"
#include "geom_object.h"
#include "problems.h"
#include "mincost_maxflow.tpp"

template class Flow<int>;
template class Flow<double>;

using namespace std;

struct Menu
{
    enum Mode{DEFAULT=0, HELP, DEFICIT, FLOW, DEFFLOW, ALL};

    string folder1 = "data/mnist/train";
    string folder2 = "data/mnist/test";
    pair<int, int> size = {28, 28};
    int N = 10;
    int range = -1;
    string out = "NO_OUTFILE_IS_GIVEN";
    Mode myMode=DEFAULT;
};

const char* helpmessage="Description:\n   Min cost matching of the two picture sets\n"
"usage: main [-h] [-size] [-folder1] [-folder2] [-N] [-range] [-out] \n\n"
"Example: ./bin/main -size 28,28 -folder1 data/mnist/train"
" -folder2 data/mnist/test -N 10\n"
"Compile with: g++ -O3 -std=c++17 -o bin/main src/main.cpp -lstdc++fs \n";
            
Menu *help(vector<string> argv)
{
    Menu *m = new Menu();
    for (int i = 0; i < argv.size(); i++)
    {
        //cout<<(string) argv[i]<<endl;
        if ((string)argv[i] == "-h" || (string)argv[i] == "--help")
        {
            cout <<helpmessage;
            m->myMode=Menu::HELP;
        }
        else if ((string)argv[i] == "--ini")
        {
            fstream inifile(argv[++i]);
            argv.resize(0);

            string temp;
            while (inifile>>temp)
            {
                argv.push_back(temp);
            }
            return help(argv);
        }
        else if ((string)argv[i] == "-size")
        {
            stringstream ss(argv[++i]);
            string first, second;
            getline(ss, first, ',');
            getline(ss, second, ',');
            m->size = {stoi(first), stoi(second)};
        }
        else if ((string)argv[i] == "-folder1")m->folder1 = argv[++i];
        else if ((string)argv[i] == "-folder2")m->folder2 = argv[++i];
        else if ((string)argv[i] == "-N")m->N = stoi(argv[++i]);
        else if ((string)argv[i] == "-range")m->range = stoi(argv[++i]);
        else if ((string)argv[i] == "-out")m->out = argv[++i];
        else if ((string)argv[i] == "-deficit")m->myMode=Menu::DEFICIT;
        else if (argv[i] == "-flow")    m->myMode=Menu::FLOW;
        else if (argv[i] == "-defFlow") m->myMode=Menu::DEFFLOW;
        else if (argv[i] == "-all") m->myMode=Menu::ALL;
    }
    return m;
}

/**
 * Description:
 *     Returns the perfect matching between the two picture dataset
 * Parameters:
 *     N:                        size of the train/test dataset
 *     size:                     size of the images
 *     train_folder/test_folder: the folder, where the datasets are
 */  
double match_mnist(int N,string folder2)
{
    Hungarian_method method = Hungarian_method();
    //std::cout << "The matching between the pictures:\n";
    string match_outfile=folder2+"/../mnist_result_"+to_string(N)+".txt";
    double result = method.run("./mnist_mtx.txt", match_outfile,N);
    return result/N;
}

double deficit(int i, int N){
    Hungarian_method method = Hungarian_method();
    method.read_mtx("./mnist_mtx.txt",N);
    method.init();

    method.sparse_all(i);
    auto p=method.Gamma();
    return ((double) p.first/p.second)*100.0;
}

pair<double,double> deficitFlow(int i,int N){
    Hungarian_method method = Hungarian_method();
    method.read_mtx("./mnist_mtx.txt",N);
    method.init();

    method.sparse_all(i);
    vector<vector<long double> > mtx=method.get_mtx();
    Flow<long double> f;
    auto ret=f.minCost_maxMatching_flow(mtx);
    return {((double)ret.first)/N,ret.second};
}

double flowMatching(int N){
    vector<vector<double> > mtx;
    beolv<double>(mtx,"./mnist_mtx.txt",N);
    Flow<double> f;
    return ((double) f.minCost_maxMatching_flow(mtx).first)/N;
}

// ====================================================================
//                                MAIN
// ====================================================================
int main(int argc, char *argv[])
{
    //flowMatching(10);
    cout << "===== MIN COST MATCHING =====" << endl;
    vector<string> argv_;
    ofstream myfile;

    for (int i = 0; i < argc; i++)
        argv_.push_back((string)argv[i]);
    
    Menu *m = help(argv_);
    if (m->myMode==Menu::HELP)
        return 1;
    else if(m->myMode==Menu::ALL){
        generate_graph(m->N,m->size, m->folder1, m->folder2);
        for(int i=1;i<10;i++){
            match_mnist(m->range*i,m->folder2);
        }
        flowMatching(m->N);
    }
    else if(m->myMode==Menu::DEFICIT || m->myMode==Menu::DEFFLOW){
        generate_graph(m->N,m->size, m->folder1, m->folder2);

        myfile.open(m->out);
        int by=1;
        int until=40;
        myfile<<by<<" "<<until<<" "<<by<<endl; // [begin end range_by]

        vector<double> flow;
        for(int i=by;i<=until;i+=by){
            cout<<"\r"<<i;
            std::cout.flush();
            if(m->myMode==Menu::DEFICIT) myfile << deficit(i, m->N)<<" ";
            else if(m->myMode==Menu::DEFFLOW){
                pair<double,double> ret=deficitFlow(i,m->N);
                myfile<<ret.first<<" ";
                flow.push_back(ret.second);
            }
        }
        cout<<endl;

        // ===== Print the amount of the flow if neccessary =====
        myfile<<endl;
        for(auto val: flow) myfile<<val<<" ";
    }
    else if (m->myMode==Menu::FLOW || m->myMode==Menu::DEFAULT)
    {
        cout<<"Matching : \n";
        generate_graph(m->N,m->size, m->folder1, m->folder2);
        myfile.open(m->out);
        myfile<<m->range<<" "<<m->N<<" "<<m->range<<endl;
        
        for (int i = 1; m->range * i <= m->N; i++)
        {
            //generate_graph(m->range*i,m->size, m->folder1, m->folder2);
            if(m->myMode==Menu::FLOW) myfile<<flowMatching(m->range*i)<<" ";
            else myfile << match_mnist(m->range*i,m->folder2) << " ";
        }
    }
    else
    {
        generate_graph(m->N,m->size, m->folder1, m->folder2);
        match_mnist(m->N,m->folder2);
    }

    myfile.close();
    delete m;


    //cout<<"===== Min Cost Perf Matching with Mincostflow ====="<<endl;
    

    return 0;
}

// ====================================================================
//                         TRASH AND UNUSED CODES
// ====================================================================
void test()
{
    pair<int, int> size(2, 2);
    Picture picture1(size);
    Picture picture2(size);

    Koord k1(1, -1);
    Koord k2(2, -1);
    Koord k3(3, -1);
    Koord k4(4, -1);

    k1.index = 0;
    k2.index = 1;
    k3.index = 2;
    k4.index = 3;

    picture1.add_point(k1);
    picture1.add_point(k2);
    picture1.add_point(k3);
    picture1.add_point(k4);
    picture2.add_point(k1);
    picture2.add_point(k2);
    picture2.add_point(k3);
    picture2.add_point(k1);

    Geometric_object::print_mtx(picture1, picture2, "data/trashexample1.txt");
    Geometric_object::print_dat(picture1, picture2, "data/trashexample1.dat");
}
