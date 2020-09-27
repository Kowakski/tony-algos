#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <vector>

using namespace std;

 void split(std::string& s, std::string& delim,std::vector< std::string >* ret)  
{  
    size_t last = 0;  
    size_t index=s.find_first_of(delim,last);  
    while (index!=std::string::npos)  
    {  
        ret->push_back(s.substr(last,index-last));  
        last=index+1;  
        index=s.find_first_of(delim,last);  
    }  
    if (index-last>0)  
    {  
        ret->push_back(s.substr(last,index-last));  
    }  
}  

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  v.clear();
  while(std::string::npos != pos2)
  {
    v.push_back(s.substr(pos1, pos2-pos1));
 
    pos1 = pos2 + c.size();
    pos2 = s.find(c, pos1);
  }
  if(pos1 != s.length())
    v.push_back(s.substr(pos1));
}

int main(){
    ifstream fin;
    fin.open("/home/sln/share/datas/Walking2/groundtruth_rect.txt");
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
        // ++count;
        // cout<<count<<":"<<stringitem<<endl;
        cout<< stringitem<<endl;
        vector < string > temp(10);
        stringitem.pop_back();
        SplitString( stringitem, temp, "\t" );
        for( int i = 0; i < temp.size(); i++ ){
            cout << temp[i]<<" ";
        }
        cout << endl;

        getline(fin, stringitem,'\n');
    }
    cout<<"Done"<<endl;

    fin.close();
    return 0;
}

