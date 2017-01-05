#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

int main()
{
	float a, b;
	cin >> a >> b;
	a = a + b;
	b = a - b;
	a = a - b;
	cout << "a=" << a << "b=" << b << endl;
	return 0;
}
