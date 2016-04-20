//
//  main.cpp
//  atoi
//
//  Created by 丁涵宇 on 4/19/16.
//  Copyright © 2016 丁涵宇. All rights reserved.
//

#include <iostream>
#include <string>   
using namespace std;

int main(int argc, const char * argv[]) {
    string str = "";
    cin >> str;
    bool isNeg = false;
    unsigned long long ret = 0;
    for (size_t i = 0; i < str.length(); i++) {
        if (str[i] != ' ') {
            str = str.substr(i);
            break;
        }
    }
    if (*(str.begin())=='+' || *(str.begin())=='-') {
        if (!isdigit(*(++str.begin()))) return 0;
        if (*(str.begin())=='-') isNeg = true;
        str = str.substr(1);
    }
    auto i = str.begin();
    while (isdigit(*i)) {
        ret = ret*10 + *i - 48;
        if (ret > (unsigned long long)INT_MAX) {
            return isNeg ? INT_MIN:INT_MAX;
        }
        i++;
    }
    return isNeg ? -(int)ret:(int)ret;
}
