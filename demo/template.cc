#include "template.h"

template<typename Real>
Test<Real>::Test(Real t)
{
	this->data = t;
	std::cout << "Test Constructor " << t << std::endl;
}

template class Test<float>;
template class Test<double>;
