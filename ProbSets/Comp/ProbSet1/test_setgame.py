# test_setgame.py

import setgame as s

from itertools import combinations

import pytest

@pytest.fixture
def set_up_cards():
    cards_1 = ["0000","0001","0002","0010",
               "0011","0012","0020","0021",
               "0022","0100","0101","0102"]
    cards_2 = ["1000","1001","1002","1010",
               "1011","1012","1020","1021",
               "1022","1100","1101","1102"]
    cards_3 = ["2000","2001","2002","2010",
               "2011","2012","2020","2021",
               "2022","2100","2101","2102"]
    return cards_1, cards_2, cards_3


def test_count_sets_ve1():
    with pytest.raises(ValueError) as excinfo:
        s.count_sets(["0000"])
    assert excinfo.value.args[0] == "Please enter a list of exactly 12 unique cards, each with 4 digits, each of which being either 0, 1, or 2."
    
def test_count_sets_ve2(set_up_cards):
    cards_1, cards_2, cards_3 = set_up_cards   
    with pytest.raises(ValueError) as excinfo:
        s.count_sets(["0000","0000","0002","0010",
               "0011","0012","0020","0021",
               "0022","0100","0101","0102"])
    assert excinfo.value.args[0] == "Please enter a list of exactly 12 unique cards, each with 4 digits, each of which being either 0, 1, or 2."
    
def test_count_sets_ve3(set_up_cards):
    cards_1, cards_2, cards_3 = set_up_cards
    with pytest.raises(ValueError) as excinfo:
        s.count_sets(["000","001","002","000",
               "001","12","000","001",
               "022","000","0101","00"])
    assert excinfo.value.args[0] == "Please enter a list of exactly 12 unique cards, each with 4 digits, each of which being either 0, 1, or 2."
    
def test_count_sets_ve4(set_up_cards):
    cards_1, cards_2, cards_3 = set_up_cards
    with pytest.raises(ValueError) as excinfo:
        s.count_sets(["5000","0001","0002","0010",
               "0011","0012","0020","0021",
               "0022","0100","0101","0102"])
    assert excinfo.value.args[0] == "Please enter a list of exactly 12 unique cards, each with 4 digits, each of which being either 0, 1, or 2."
    

def test_count_sets_counting(set_up_cards):
    cards_1, cards_2, cards_3 = set_up_cards
    sets_1 = 0
    sets_2 = 0
    sets_3 = 0
    for i in combinations(cards_1,3):
        if s.is_set(i[0],i[1],i[2]):
            sets_1+=1
    for i in combinations(cards_2,3):
        if s.is_set(i[0],i[1],i[2]):
            sets_2+=1
    for i in combinations(cards_3,3):
        if s.is_set(i[0],i[1],i[2]):
            sets_3+=1
    
    assert s.count_sets(cards_1) == sets_1, "failed at counting sets"
    assert s.count_sets(cards_2) == sets_2, "failed at counting sets"
    assert s.count_sets(cards_3) == sets_3, "failed at counting sets"
    

def test_is_set():
    abc_1 = ["1001","1001","2002"] # not a set
    abc_2 = ["1001","1001","2001"] # not a set
    abc_3 = ["0000","0001","1001"] # not a set
    isset1, isset2, isset3 = True, True, True
    for i in range(4):
        
        print("1")
        print(set([abc_1[0][i], abc_1[1][i], abc_1[2][i]]))
        if len(set([abc_1[0][i], abc_1[1][i], abc_1[2][i]])) not in [1,3]:
            isset1 = False
        
        print("2")
        print(set([abc_2[0][i], abc_2[1][i], abc_2[2][i]]))
        if len(set([abc_2[0][i], abc_2[1][i], abc_2[2][i]])) not in [1,3]:
            isset2 = False
        
        print("3")
        print(set([abc_3[0][i], abc_3[1][i], abc_3[2][i]]))
        if len(set([abc_3[0][i], abc_3[1][i], abc_3[2][i]])) not in [1,3]:
            isset3 = False
    assert s.is_set(abc_1[0],abc_1[1],abc_1[2]) == isset1, "failed at determining if a set"
    assert s.is_set(abc_2[0],abc_2[1],abc_2[2]) == isset2, "failed at determining if a set"
    assert s.is_set(abc_3[0],abc_3[1],abc_3[2]) == isset3, "failed at determining if a set"

    
    