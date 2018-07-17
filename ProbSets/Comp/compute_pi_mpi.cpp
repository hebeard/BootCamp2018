//
//  compute_pi_mpi.cpp
//  
//
//  Created by Harrison Beard on 7/17/18.
//

#include "compute_pi_mpi.hpp"
#include <iostream>
using namespace std;

const int num_steps = 500000000;

int main(void){
    int i;
    double sum = 0.0;
    double pi = 0.0;
    
    cout << "using " << omp_get_max_threads() << " OpenMP threads " << endl;
    
    const double w = 1.0/double(num_steps);
    
    double time = -omp_get_wtime();
    
    #pragma omp parallel firstprivate(sum)
    {
        #pragma omp for
        for (int i=0; i<num_steps; ++i)
        {
            double x = (i+0.5)*w;
            sum += 4.0/(1.0+x*x);
        }
        #pragma omp critical
            {
            pi = pi + w*sum;
            }
    }
    time += omp_get_wtime();
    
    cout << num_steps << " steps approximates pi as: " << pi << ", with relative error " << fabs(M_PI-pi)/M_PI << endl;
    cout << "the solution took " << time << " seconds" << endl;
}
