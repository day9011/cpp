#include "template2.h"

template <typename Real>
Test2<Real>::Test2(Real n)
{
	this->data = n;
	std::cout << "Test2 Constructor " << n;
}

template <typename Real>
void Test2<Real>::createTest()
{
	float n = 2.0;
	Test<float> *t = new Test<float>(n);
}

template class Test2<float>;
template class Test2<double>;
