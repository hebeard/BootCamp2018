//
//  pi_rand.cpp
//  
//
//  Created by Harrison Beard on 7/14/18.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
using namespace std;

int main()
{
    double niter1 = 100;
    double niter2 = 1000;
    double niter3 = 10000;
    double niter4 = 1000000000;
    double x,y;
    int i;
    int count=0;
    double z;
    double pi1;
    double pi2;
    double pi3;
    double pi4;
    //srand(time(NULL));
    //main loop
    for (i=0; i<niter1; ++i)
    {
        //get random points
        x = (double)random()/RAND_MAX;
        y = (double)random()/RAND_MAX;
        z = sqrt((x*x)+(y*y));
        //check to see if point is in unit circle
        if (z<=1)
        {
            ++count;
        }
    }
    pi1 = ((double)count/(double)niter1)*4.0;          //p = 4(m/n)
    count =0;
    for (i=0; i<niter2; ++i)
    {
        //get random points
        x = (double)random()/RAND_MAX;
        y = (double)random()/RAND_MAX;
        z = sqrt((x*x)+(y*y));
        //check to see if point is in unit circle
        if (z<=1)
        {
            ++count;
        }
    }
    pi2 = ((double)count/(double)niter2)*4.0;          //p = 4(m/n)
    count =0;
    for (i=0; i<niter3; ++i)
    {
        //get random points
        x = (double)random()/RAND_MAX;
        y = (double)random()/RAND_MAX;
        z = sqrt((x*x)+(y*y));
        //check to see if point is in unit circle
        if (z<=1)
        {
            ++count;
        }
    }
    pi3 = ((double)count/(double)niter3)*4.0;          //p = 4(m/n)
    count =0;
    for (i=0; i<niter4; ++i)
    {
        //get random points
        x = (double)random()/RAND_MAX;
        y = (double)random()/RAND_MAX;
        z = sqrt((x*x)+(y*y));
        //check to see if point is in unit circle
        if (z<=1)
        {
            ++count;
        }
    }
    pi4 = ((double)count/(double)niter4)*4.0;          //p = 4(m/n)

    cout << "•  Pi for N=100: \t\t" << pi1
    << "\n•  Pi for N=1,000: \t\t" << pi2
    << "\n•  Pi for N=10,000: \t\t" << pi3
    << "\n•  Pi for N=1,000,000,000: \t" << pi4 << endl;
    
    return 0;
}
