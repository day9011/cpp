#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

int main()
{
	int t[] = {2,4,7,1,5,9,6,3,8};
	std::nth_element(t, t + 6, t + 9);
	for (int i = 6; i < 9; i++)
		printf("\nt[%d]=%d", i, t[i]);
	printf("\n");
	int a(123);
	printf("\na=%d\n", a);
	return 0;
}
