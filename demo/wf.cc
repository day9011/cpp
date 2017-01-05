#include <fstream>

int main()
{
	std::ofstream out;
	out.open("test.txt");
	if (out.is_open())
	{
		out << "for a test.";
		out.close();
	}
	return 0;
}
