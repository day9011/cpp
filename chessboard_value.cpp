//
//  main.cpp
//  test
//
//  Created by 丁涵宇 on 4/2/16.
//  Copyright © 2016 丁涵宇. All rights reserved.
//

#include <iostream>
#include <vector>
#include <stack>

inline int max(int a, int b){
    return a > b ? a : b;
}

using namespace std;

void input_metric (vector<vector<int>> &, int, int);

int M = 0, N = 0;

int main(int argc, const char * argv[]) {
    // insert code here...
//    int test = 0;
//    char c = 0x14;
//    test = test | c;
//    cout << test << endl;
    cin >> M >> N;
    char c = getchar();
//    if (c == '\n')
//        cout << "123";
    vector<vector<int>> metric(M, vector<int>(N));
    input_metric(metric, M, N);
    vector<vector<int>> maxvalue(M, vector<int>(N));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            if (i == 0 & j == 0) {
                maxvalue[i][j] = metric[0][0];
            }
            else if(i == 0)
                maxvalue[i][j] = maxvalue[i][j - 1] + metric[i][j];
            else if(j == 0)
                maxvalue[i][j] = maxvalue[i - 1][j] + metric[i][j];
            else
                maxvalue[i][j] = max(maxvalue[i][j - 1], maxvalue[i - 1][j]) + metric[i][j];
        }
    cout << endl << maxvalue[M-1][N-1] << endl;
//    for (int i = 0;i < M; i++){
//        for (int j = 0;j < N; j++)
//            cout << metric[i][j] << ' ';
//        cout << endl;
//    }
    return 0;
}
void input_metric(vector<vector<int>> &metric, int m, int n){
    char ch;
    int enter_num = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int num = 0;
            while ((ch = getchar()) != ' '){
                if (ch == '\n') {
                    enter_num++;
                    break;
                }
                int ch_t = ch - 48;
                num = num * 10 + ch_t;
            }
            metric[i][j] = num;
            if (enter_num > m) return;
        }
    }
}
