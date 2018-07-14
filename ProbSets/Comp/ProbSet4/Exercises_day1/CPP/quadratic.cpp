#include <iostream>
#include <math.h>
using namespace std;

int main()
{
    double a;
    double b;
    double c;
    double x_1;
    double x_2;

    cout << "a:", cin >> a;
    cout << "b:", cin >> b;
    cout << "c:", cin >> c;

    x_1 = (-1 * b + pow(pow(b,2) - 4 * a * c, 0.5) ) / (2 * a);
    x_2 = (-1 * b - pow(pow(b,2) - 4 * a * c, 0.5) ) / (2 * a);

    cout << "The two roots of the quadratic " << a << "x^2 + " 
    << b << "x + c = 0 are " << x_1 << " and " << x_2 << "." << endl;

    return 0;
}