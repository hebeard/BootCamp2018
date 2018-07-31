#include <iostream>
using namespace std;
static long num_steps = 100000; double step;

int main(){
    
    int i;
    double x, pi, sum = 0.0;

    step = 1.0/(double) num_steps;

    for (i=0; i<num_steps; i++){

        x = (i+0.5) * step;
        sum = sum + 1.0 / (1.0 + x * x); // only produces one-fourth of pi
    }
    
    pi = step * sum;

    cout << "Pi over four: " << pi << endl;

    return 0;
}