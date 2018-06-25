# test_fraction.py

import fraction as fr

import pytest

@pytest.fixture
def set_up_fractions():
    frac_1_3 = fr.Fraction(1, 3)
    frac_1_2 = fr.Fraction(1, 2)
    frac_n2_3 = fr.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = fr.Fraction(30, 42) # 30/42 reduces to 5/7.
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(ZeroDivisionError) as excinfo:
        fr.Fraction(1,0)
    assert excinfo.value.args[0] == "denominator cannot be zero"
    with pytest.raises(TypeError) as excinfo:
        fr.Fraction(1,"2")
    assert excinfo.value.args[0] == "numerator and denominator must be integers"
    
def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1 / 3"
    assert str(frac_1_2) == "1 / 2"
    assert str(frac_n2_3) == "-2 / 3"
    assert str(fr.Fraction(2,1)) == "2"
    
def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.
    assert float(frac_1_3) == frac_1_3.numer / frac_1_3.denom
    
def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == fr.Fraction(1, 2)
    assert frac_1_3 == fr.Fraction(2, 6)
    assert frac_n2_3 == fr.Fraction(8, -12)
    other = fr.Fraction(1, 2)
    assert (frac_1_2 == other) == (frac_1_2.numer == other.numer) and \
           (frac_1_2.denom == other.denom)
    assert (frac_1_2 == .5) == (float(frac_1_2) == .5)
    
def test_fraction_add(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 + frac_1_3 == fr.Fraction(frac_1_2.numer * frac_1_3.denom + \
                                              frac_1_2.denom * frac_1_3.numer, \
                                              frac_1_2.denom * frac_1_3.denom)
    assert frac_n2_3 + frac_1_2 == fr.Fraction(frac_n2_3.numer * frac_1_2.denom + \
                                              frac_n2_3.denom * frac_1_2.numer, \
                                              frac_n2_3.denom * frac_1_2.denom)
    
def test_fraction_sub(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 - frac_1_3 == fr.Fraction(frac_1_2.numer * frac_1_3.denom - \
                                              frac_1_2.denom * frac_1_3.numer, \
                                              frac_1_2.denom * frac_1_3.denom)
    assert frac_n2_3 - frac_1_2 == fr.Fraction(frac_n2_3.numer * frac_1_2.denom - \
                                              frac_n2_3.denom * frac_1_2.numer, \
                                              frac_n2_3.denom * frac_1_2.denom)
    
def test_fraction_mul(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 * frac_1_3 == fr.Fraction(frac_1_2.numer * frac_1_3.numer, \
                                              frac_1_2.denom * frac_1_3.denom)
    assert frac_n2_3 * frac_1_2 == fr.Fraction(frac_n2_3.numer * frac_1_2.numer, \
                                              frac_n2_3.denom * frac_1_2.denom)

def test_fraction_truediv(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 / frac_1_3 == fr.Fraction(frac_1_2.numer * frac_1_3.denom, \
                                              frac_1_2.denom * frac_1_3.numer)
    assert frac_n2_3 / frac_1_2 == fr.Fraction(frac_n2_3.numer * frac_1_2.denom, \
                                              frac_n2_3.denom * frac_1_2.numer)
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_1_3 / fr.Fraction(0,1)
    assert excinfo.value.args[0] == "cannot divide by zero"
