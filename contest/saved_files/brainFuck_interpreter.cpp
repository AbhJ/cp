#include <map>
#include <set>
#include <list>
#include <cmath>
#include <ctime>
#include <deque>
#include <queue>
#include <stack>
#include <string>
#include <bitset>
#include <cstdio>
#include <limits>
#include <vector>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>

using namespace std;

// GLOBAL VARIABLE SO, ALREADY INITIALISED TO 0
char arr[30000];

// d IS DATA POINTER
// i IS POINTER THROUGH SOURCE CODE STRING
// cnt IS COUNT OF BRACKETS (it could have done by stack, but I didn't have much time left)

int d, i, cnt;

string str;

// FUNCTIONS TO MOVE DATA POINTER 
void pointer_increment(){
    // >
    d++;
}
void pointer_decrement(){
    // <
    d--;
}

// FUNCTIONS TO CHANGE DATA AT POINTED LOCATIONS
void data_increment(){
    // +
    arr[d]++;
}
void data_decrement(){
    // -
    arr[d]--;
}

// I/O FUNCTIONS
void output_data(){
    // .
    cout << arr[d];
    cout.flush();
}
void input_data(){
    // ,
    cin >> arr[d];
}

// SQUARE BRACKETS
void jump_forward(){
   if(arr[d] == 0){
       cnt++;
       while(str[i] != ']' or cnt != 0){
           i++;
           cnt += (str[i] == '[');
           cnt -= (str[i] == ']');
       }
   }
}
void jump_backward(){
   if(arr[d] != 0){
       cnt++;
       while(str[i] != '[' or cnt != 0){
           i--;
           cnt -= (str[i] == '[');
           cnt += (str[i] == ']');
       }
   }
}

int main(int argc, char **argv) {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    cin >> str;
    for(i = 0; i < str.length(); i++){
        switch(str[i]){
            case '>':
                pointer_increment();
                break;
            case '<':
                pointer_decrement();
                break;
            case '+':
                data_increment();
                break;
            case '-':
                data_decrement();
                break;
            case '.':
                output_data();
                break;
            case ',':
                input_data();
                break;
            case '[':
                jump_forward();
                break;
            case ']':
                jump_backward();
                break;
            default:
                cout << "Error";
        }
    }
    return 0;
}
