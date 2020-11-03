#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>

using namespace std;
int main()
{
    ifstream fin;
    fin.open("./groundtruth_rect.txt");
    if (fin.is_open() == false)
    {
        cerr<<"can not open"<<endl;
        exit(EXIT_FAILURE);
    }
    string stringitem;
    int count = 0;
    getline(fin,stringitem,'\n');
    while (fin)
    {
        ++count;
        cout<<count<<":"<<stringitem<<endl;
        getline(fin,stringitem,'\n');
    }
    cout<<"Done"<<endl;

    fin.close();
    return 0;
}

