#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <vector>
#include <cstdlib> //added
#include <cstring> //added

using namespace std;

int main(int argc, char* argv[]){

if(argc != 3){
	cout<<"USAGE: ./hmmsort input_filename output_filename"<<endl;
	exit(0);
}

multimap<int, int> len;
vector<string> dbstr;
vector<string> desc;
ifstream input;
input.open(argv[1], ios::binary);
input.seekg(0, ios::end);

long length = input.tellg();
//cout<<"Length:"<<length<<endl;
char* buffer = new char[length];
input.seekg(0, ios::beg);
input.read(buffer, length);
input.close();
int i = 0; 

char*  pch = strtok (buffer,"\n");
	if(pch != NULL){
		desc.push_back(string(pch));
		pch = strtok(NULL,">");
	}
  
  while (pch != NULL)
  {
	//len.push_back(strlen(pch));
	len.insert(make_pair(strlen(pch),i));
	i++;
	dbstr.push_back(string(pch));
	//cout<<pch<<endl;
//	cout<<i<<endl;
        pch = strtok (NULL,"\n");
	if(pch != NULL){
		desc.push_back(string(pch));
		pch = strtok(NULL,">");
	}  

  }
 //cout<<len.size()<<"	"<<dbstr.size()<<"	"<<desc.size()<<endl;

ofstream output;
//string file = "sort_" + string(argv[1]);
string file = string(argv[2]);
output.open(file.c_str());
//multimap<int, int>::reverse_iterator iter;
multimap<int, int>:: iterator iter;

for(iter = len.begin(); iter != len.end(); iter++){
	output<<">"<<desc[iter->second]<<endl;
	output<<dbstr[iter->second];

}
output.close();
}
