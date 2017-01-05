#include "cppHeader.h"

using namespace std;

int main(int argc, char *argv[]) {
	string t(argv[1]);
	int a = atoi(t.c_str());
	cout << "cpp file:" << endl; 
	printinc(a);
	cout << endl;
	return 0;
}
