# test_smallest_factor.py

import smallest_factor as sf

def test_evenprime():
    assert sf.smallest_factor(2) == 2, "failed on even prime"
def test_smallprimes():
    assert sf.smallest_factor(3) == 3, "failed on small primes"
def test_largeprimes():
    assert sf.smallest_factor(105541) == 105541, "failed on large primes"
def test_squares():
    assert sf.smallest_factor(4) == 2, "failed on square"
def test_composites():
    assert sf.smallest_factor(6) == 2, "failed on non-square composite"
def test_highlycomposite():
    assert sf.smallest_factor(840) == 2, "failed on highly composite number"