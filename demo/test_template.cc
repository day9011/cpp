#include "template2.h"

int main()
{
	float num = 1.0;
	Test2<float> *t = new Test2<float>(num);
	t->createTest();
}
