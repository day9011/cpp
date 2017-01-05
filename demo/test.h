#include <stdio.h>

class Base
{
	public:
		Base() {
			printf("in base\n");
		}
		virtual void printName() { printf("my name is Base\n"); }
		virtual void printFunc() = 0;
		~Base() {}
	private:
		Base(const Base&);
		Base& operator = (const Base&);
};

class Child : public Base
{
	public:
		Child() {
			printf("in child\n");
		}

		virtual void printFunc() { printf("for a test\n"); }

		~Child() {}
};
