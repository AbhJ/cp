
// LIBRARY_CODE
#include "bits/stdc++.h"
using namespace std;
// THOUGH STL LIBRARY FOR INSERTION SORT IS NOT THERE / NOT ALLOWED HERE, WE CAN USE THE STL TO IMPLEMENT IT

template<typename stdContainer>
void insertion_sort(stdContainer &container){
    // HERE stdContainer IS A GENERIC STANDARD C++ STL CONTAINER
    
    for(auto it = container.begin(); it != container.end(); it++){
        // FINDING CORRECT POSITION FOR it
        auto const index = upper_bound(container.begin(), it, *it);
        
        // NOW WE HAVE TO KEEP ALL SMALLER ELEMENTS TO THE LEFT OF it
        // THIS CAN BE EASILY DONE USING STL rotate FUNCTION
        rotate(index, it, it + 1);
        
        // WE COULD HAVE MOVE IT USING A POINTER BUT THE CODE WOULD BE LONGER
        // AND LESS READABLE.
    }
}

// TIME = O(n * n) SPACE = O(n)

// APPLICATION_CODE
#include <iostream>
#include <vector> 

int main(  ) {
    std::vector<int> input;
    int x;
    while( std::cin >> x ) {
        input.push_back( x );
    }

    insertion_sort( input );
        
    for( auto & x: input ) {
        std::cout << x << std::endl;
    }
}
