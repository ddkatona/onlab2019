#pragma once

#include <iostream>
#include <vector> 
#include <random>
#include <math.h>
#include <fstream>
#include <time.h>
#include <limits>
#include <algorithm>
#include <chrono>
#include <experimental/filesystem>
//#include <stdlib.h>

using namespace std;
namespace fs = std::experimental::filesystem;

// =======================================================================
//                                  KOORD
// =======================================================================
struct Koord{
    double x,y;
    double phi;
    int index;
    Koord(double x0=0, double y0=0, double phi=0): 
        x(x0),y(y0), phi(phi){}

    static bool third_coord(Koord& k1, Koord& k2){
        return k1.phi<k2.phi;
    }
};
ostream& operator<<(ostream& os,Koord& k){
    os<<k.x<<" "<<k.y;
}

double euclidian_dist(Koord& k1, Koord& k2){
    return sqrt((k1.x-k2.x)*(k1.x-k2.x)+(k1.y-k2.y)*(k1.y-k2.y));
}
// =======================================================================
//                             GEOMETRIC OBJECT
// =======================================================================
struct Geometric_object{
public:
    Geometric_object(){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator=std::default_random_engine (seed);
        //cout<<"Random nums: "<<seed<<endl;
        distribution=std::uniform_real_distribution<double>(0.0,1.0);
    }

    virtual Koord get_random_point(double rand)=0;
    virtual double dist(Koord& k1, Koord& k2)=0;

    void add_point(Koord point){
        points.push_back(point);
    }

    void generate(int N){

        for(int i=0;i<N;i++){
            double r=distribution(generator);
            add_point(get_random_point(r));
        }
    }

    void write(string filename="out.txt"){
        ofstream file;
        file.open(filename.c_str());
        for(int i=0;i<points.size();i++){
            file<<points[i]<<endl;
        }
    }

    // Writes the the given matching to the given file
    void static write(
        vector<pair<int,int> >& v,
        Geometric_object &o1,
        Geometric_object &o2,
        string filename="out.txt")
    {

        ofstream file;
        file.open(filename.c_str());
        for(int i=0;i<v.size();i++){
            int u0=v[i].first;
            int v0=v[i].second;
            //file<<o1.points[u0]<<" "<<o2.points[v0]<<endl;
            file<<u0<<" "<<v0<<endl;
        }
    }

    // Description:
    //    --> Print incidence mtx of the two object for GLPK-SOL
    // Parameters:
    //    --> Geometric object1
    //    --> Geometric object2
    //    --> filename of the output mtx file
    void static print_dat(Geometric_object &o1, Geometric_object &o2, std::string fname){
        ofstream myfile_dat(fname.c_str());
        myfile_dat<<"data;\n";
        
        myfile_dat<<"set I:=";
        for(int i=0;i<o1.points.size();i++) myfile_dat<<i<<" ";
        myfile_dat<<";\n";

        myfile_dat<<"set J:=";
        for(int i=0;i<o2.points.size();i++) myfile_dat<<i<<" ";
        myfile_dat<<";\n";

        myfile_dat<<"param c:";
        for(int i=0;i<o2.points.size();i++) myfile_dat<<i<<" ";
        myfile_dat<<":=\n";

        for(int i=0;i<o1.points.size();i++){
            myfile_dat<<i<<" ";
            for(int j=0;j<o2.points.size();j++){
                long double dist= o1.dist(o1.points[i],o2.points[j]);
                myfile_dat<<dist<<" ";
            }
            myfile_dat<<endl;
        }
        myfile_dat<<";\nend;";


    }

    void static print_mtx(Geometric_object &o1, Geometric_object &o2, std::string fname, bool std_out=false){
        ofstream myfile(fname.c_str());

        myfile<<o1.points.size()<<" "<<o2.points.size()<<endl;
        if(std_out) cout<<o1.points.size()<<" "<<o2.points.size()<<endl;
        

        for(int ind=0;ind < o1.points.size();ind++){
            Koord p1=o1.points[ind];
            if(std_out) cout<<ind<<" ";
            myfile<<ind<<" ";
            for(int j=0;j<o2.points.size();j++){
                Koord p2=o2.points[j];
                long double dist= o1.dist(p1,p2);
                myfile<<dist<<" ";
                if(std_out) cout<<dist<<" ";
            }
            myfile<<endl;
            if(std_out) cout<<endl;
        }
        myfile.close();
    }
public:
    vector<Koord> points;

private:
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;
    
};

// =======================================================================
//                               CIRCLE
// =======================================================================

struct Circle: Geometric_object{
    Koord center;
    double radius;
    vector<double> phis;

    Circle(Koord c, double r):center(c),radius(r){}

    virtual Koord get_random_point(double rand0){
        double phi=rand0*M_PI*2;
        double x=center.x+radius*cos(phi);
        double y=center.y+radius*sin(phi);

        return Koord(x,y,phi);
    }

    virtual double dist(Koord& k1, Koord& k2){
        return euclidian_dist(k1,k2);
    }
    
    vector<pair<int,int> > static sort(Circle& c1, Circle& c2){
        double min_sum=std::numeric_limits<double>::max();
        int ind=0;
        int N=c1.points.size();

        vector<Koord> copy1,copy2;
        for(int i=0;i<c1.points.size();i++){
            Koord u=c1.points[i];
            u.index=i;
            copy1.push_back(u);
        }
        for(int i=0;i<c2.points.size();i++){
            Koord u=c2.points[i];
            u.index=i;
            copy2.push_back(u);
        }

        // ===== Sort the elements after nail =====
        std::sort(copy1.begin(),copy1.end(), Koord::third_coord);
        std::sort(copy2.begin(),copy2.end(), Koord::third_coord);

        // ===== Try first node to every other node, and choose minimum =====
        for(int i=0;i<copy1.size();i++){
            double sum=0.0;
            for(int j=0;j<copy2.size();j++){
                sum+=c1.dist(copy1[(i+j)%N], copy2[j%N]);
            }
            if(sum<min_sum){
                min_sum=sum;
                ind=i;
            }
            //std::cout<<"sum: "<<sum<<endl;
        }

        vector<pair<int,int> > v;
        double sum=0.0;
        for(int i=0;i<c1.points.size();i++){
            int u0=(i+ind)%N;
            int v0=i;
            sum+=c1.dist(copy1[u0], copy2[v0]);
            //cout<<"Sort: "<<u0<<" "<<v0<<endl;
            v.push_back({copy1[u0].index,copy2[v0].index});
        }
        std::cout<<"Heuristic, Min sum: "<<sum<<endl;
        return v;
    }
    
};


void generate_Circles(int N, bool write_to_file = false, string filename = "data/mtx_circles")
{
    srand(time(0));
    //srand (5);
    // ===== First Cicle: =====
    Circle c1(Koord(0, 0), 2);
    c1.generate(N);

    // ===== Second Cicle: =====
    Circle c2(Koord(0, 0), 1);
    c2.generate(N);

    // ===== Write to file =====
    if (write_to_file)
    {
        c1.write("data/circles01.txt");
        c2.write("data/circles02.txt");
    }

    // Generate incidence mtx:
    Geometric_object::print_mtx(c1, c2, filename + ".txt");
    Geometric_object::print_dat(c1, c2, filename + ".dat");

    // Heuristic:
    //vector<pair<int,int> > sorted=Circle::sort(c1,c2);
    //Geometric_object::write(sorted,c1, c2, "data/HEUcircles_sorted.txt");
}

void HUN_for_circles(string filename = "data/mtx_circles", string output = "data/HUNcircles_sorted.txt")
{
    Hungarian_method method = Hungarian_method();
    method.read_mtx(filename + ".txt",100);
    method.init();
    method.alternating_path();
    method.print_matching(output);
}

// =======================================================================
//                                 LINE
// =======================================================================
struct Line: Geometric_object{
    Koord a,b;

    Line(Koord a, Koord b):a(a),b(b){}

    virtual Koord get_random_point(double rand0){
        //double phi=((double) rand() / (RAND_MAX))*M_PI*2;
        double x=rand0*a.x+(1-rand0)*b.x;
        double y=rand0*a.y+(1-rand0)*b.y;

        return Koord(x,y);
    }

    virtual double dist(Koord& k1, Koord& k2){
        return euclidian_dist(k1,k2);
    }
    
};
void generate_Lines(int N, bool write_to_file = false)
{
    srand(time(0));
    //srand (5);
    // ===== First Line: =====
    Line l1(Koord(0, 0), Koord(1, 1));
    l1.generate(N);

    // ===== Second Line: =====
    Line l2(Koord(0, 1), Koord(2, 4));
    l2.generate(N);

    if (write_to_file)
    {
        l1.write("data/lines01.txt");
        l2.write("data/lines02.txt");
    }
}


// =======================================================================
//                               PICTURE
// =======================================================================

struct Picture: Geometric_object{

    Picture(pair<int,int> &size): N(size.first),M(size.second){}

    virtual Koord get_random_point(double rand0){
        return {-1,-1};
    }

    virtual double dist(Koord& k1, Koord& k2){
        int man_dist=abs((k1.index/M)-(k2.index/M))+abs((k1.index%M)-(k2.index%M));
        if(man_dist==0) return (abs(k1.x-k2.x));
        //else if(man_dist>10.0) return 999999.9;
        else return (abs(k1.x-k2.x))*sqrt((double) man_dist);
        //else return (abs(k1.x-k2.x))*(man_dist);
    }

private:
    int N,M;
    
};

// ==================================================================
//                        HUN WITH PICTURES
// ==================================================================
double wasserstein_dist(vector<vector<double>> &, vector<vector<double>> &);
vector<vector<vector<double>>> train0;
vector<vector<vector<double>>> test0;

void read_picture(std::string filename, vector<vector<double>> &pict, pair<int, int> size)
{
    ifstream myfile(filename.c_str());
    int N = size.first;
    int M = size.second;

    pict.resize(N, vector<double>(M, 0.0));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            myfile >> pict[i][j];
        }
    }
}

double euclidian_dist(vector<vector<double>> &p1, vector<vector<double>> &p2)
{
    int n = p1.size();
    assert(p1.size()!=0);
    int m = p1[0].size();
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            sum += abs(p1[i][j] - p2[i][j]) * abs(p1[i][j] - p2[i][j]);
        }
    }
    return sqrt(sum);
}

double wasserstein_dist(vector<vector<double>> &p1, vector<vector<double>> &p2)
{
    assert(p1.size()!=0);
    pair<int, int> size(p1.size(), p1[0].size());

    Picture picture1(size);
    Picture picture2(size);

    vector<int> v1, v2;

    for (int i = 0; i < size.first; i++)
    {
        for (int j = 0; j < size.second; j++)
        {
            Koord k = Koord(p1[i][j], -1);
            k.index = i * size.first + j;
            picture1.add_point(k);
            v1.push_back(p1[i][j]);

            Koord k2 = Koord(p2[i][j], -1);
            k2.index = i * size.first + j;
            picture2.add_point(k2);
            v2.push_back(p2[i][j]);
        }
    }

    // ===== Generate incidence mtx: =====
    // TODO: not from file
    Geometric_object::print_mtx(picture1, picture2, "data/mtx_pictures.txt");
    Geometric_object::print_dat(picture1, picture2, "data/mtx_pictures.dat");

    Hungarian_method method = Hungarian_method();
    double result = method.run("data/mtx_pictures.txt", "data/HUNpictures_sorted.txt",size.first);

    return result;
}

/**
 * Description:
 *     Creates the Incidence graph to a temp file
 * Parameters:
 *     N:                        size of the train/test dataset
 *     size:                     size of the images
 *     train_folder/test_folder: the folder, where the datasets are
 */  
void generate_graph(int N,
                   pair<int, int> &size,
                   string& train_folder,
                   string& test_folder){

    // TODO: for small dataset save into memory instead of hard disc
    // ===== Create temporary output file =====
    std::string filename = "./mnist_mtx.txt";
    ofstream myfile(filename.c_str());
    myfile << N << " " << N << endl;

    int n = size.first;
    int m = size.second;

    // ===== Read test, and train datasets from file =====
    vector<vector<double>> p1, p2;
    int iterator = 0;
    string path;
    for (const auto &entry : fs::directory_iterator(train_folder))
    {
        //read_picture(entry.path(), p1, {n, m});
        string path=train_folder+"/image_"+to_string(iterator)+".txt";
        //cout<<path<<endl;
        read_picture(path, p1, {n, m});
        train0.push_back(p1);
        if ((++iterator) > N)
            break;
    }
    iterator = 0;
    for (const auto &entry : fs::directory_iterator(test_folder))
    {
        //read_picture(entry.path(), p2, {n, m});
        string path=test_folder+"/image_"+to_string(iterator)+".txt";
        //cout<<path<<endl;
        read_picture(path, p2, {n, m});
        test0.push_back(p2);
        if (++iterator > N)
            break;
    }

    // We suppose, that the data is shuffled (during the generation)
    //     If not: 
    //srand(0); std::random_shuffle ( train0.begin(), train0.end() );
    //srand(1); std::random_shuffle ( test0.begin(), test0.end() );
    // ===== Generate the Incidence matrix =====
    for (int i = 0; i < N; i++)
    {
        myfile << i << " ";
        for (int j = 0; j < N; j++)
        {
            // TODO: just if we match the dataset with itself
            if (i == j && 0) myfile << "99999999"<< " ";
            else
            {
                int pict_dist = 1;
                switch (pict_dist)
                {
                case 0: // ===== Matching with Wasserstein distance between pictures =====
                    myfile << wasserstein_dist(train0[i], test0[j]) << " ";
                case 1: // ===== Matching with Euclidean distance between pictures =====
                    myfile << euclidian_dist(train0[i], test0[j]) << " ";
                }
            }
        }
        myfile << endl;
    }
    myfile.close();
}




