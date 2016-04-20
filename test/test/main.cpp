//
//  main.cpp
//  test
//
//  Created by 丁涵宇 on 4/19/16.
//  Copyright © 2016 丁涵宇. All rights reserved.
//

#include <iostream>
#include <string>
#include <stack>

using namespace std;

char swap_case(char c){
    if (c <= 'z' && c >= 'a')return (c-32);
        else return (c+32);
}

int main(int argc, const char * argv[]) {
    // insert code here...
    int n = 500;
    char c[n];
    cin.get(c, n);
    string str(c);
    size_t j = 0;
    size_t i = 0;
    stack<string> str_stack;
    bool isstr = false;
    for (; i < str.length(); i++) {
        if (str[i] == ' ') {
            if (isstr) {
                str_stack.push(str.substr(j, i - j));
                isstr = !isstr;
            }
            continue;
        }
        if ((str[i] <= 'z' && str[i] >= 'a') || (str[i] <= 'Z' && str[i] >= 'A')) {
            if (!isstr) {
                j = i;
                isstr = !isstr;
            }
            str[i] = swap_case(str[i]);
        }
    }
    if (isstr) {
        str_stack.push(str.substr(j, i - j));
    }
    cout << str_stack.size() << endl;
    string ret = "";
    while (!str_stack.empty()) {
        ret += str_stack.top();
        if (str_stack.size() > 1) {
            ret += ' ';
        }
        str_stack.pop();
    }
    cout << ret << endl;
    return 0;
}
