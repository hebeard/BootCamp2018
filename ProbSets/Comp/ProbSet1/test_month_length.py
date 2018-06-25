# test_month_length.py

import month_length as ml

def test_30DayNonLeap():
    assert ml.month_length("April") == 30, "failed on 30-day month "+\
                                           " on a non-leap year"
def test_31DayNonLeap():
    assert ml.month_length("January") == 31, "failed on 31-day month "+\
                                             " on a non-leap year"
    
def test_FebNonLeap():
    assert ml.month_length("February") == 28, "failed on Februrary "+\
                                               " on a non-leap year"

def test_30DayLeap():
    assert ml.month_length("April",leap_year=True) == 30, "failed on "+\
                                            "30-day month on a leap year"

def test_31DayLeap():
    assert ml.month_length("January",leap_year=True) == 31, "failed on "+\
                                           "31-day month on a leap year"
    
def test_FebLeap():
    assert ml.month_length("February",leap_year=True) == 29, "failed on "+\
                                                "Februrary on a leap year"

def test_notAMonth():
    assert ml.month_length("X") == None, "failed on something that is not a month"