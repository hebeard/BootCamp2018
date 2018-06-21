# test_smallest_factor.py

import smallest_factor

def test_smallest_factor():
    assert smallest_factor(2) == 2, "failed on even prime"
    assert smallest_factor(3) == 3, "failed on small primes"
    assert smallest_factor(4) == 2, "failed on square"