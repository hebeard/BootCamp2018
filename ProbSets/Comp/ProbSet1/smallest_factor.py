#smallest_factor.py

def smallest_factor(n):
    """Return smallest prime factor of pos int n"""
    if n==1: return 1
    for i in range(2,int(n**.5)):
        if n % i == 0 : return i
    return n