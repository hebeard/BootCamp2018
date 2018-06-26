# setgame.py

from itertools import combinations

def count_sets(cards):
    """Return the number of sets in the provided Set hand.
    Parameters:
    cards (list(str)) a list of twelve cards as 4-bit integers in
    base 3 as strings, such as ["1022", "1122", ..., "1020"].
    Returns:
    (int) The number of sets in the hand.
    Raises:
    ValueError: if the list does not contain a valid Set hand, meaning
    - there are not exactly 12 cards,
    - the cards are not all unique,
    - one or more cards does not have exactly 4 digits, or
    - one or more cards has a character other than 0, 1, or 2.
    """
    for i in cards:
        if len(i) != 4:
            raise ValueError("Please enter a list of exactly 12 unique "+\
                         "cards, each with 4 digits, each of which being either 0, 1, or 2.")
        for j in i:
            if j not in str(list(range(3))):
                raise ValueError("Please enter a list of exactly 12 unique "+\
                         "cards, each with 4 digits, each of which being either 0, 1, or 2.")
    if len(cards) != 12 or len(set(cards)) != len(cards):
        raise ValueError("Please enter a list of exactly 12 unique "+\
                         "cards, each with 4 digits, each of which being either 0, 1, or 2.")
    count = 0
    for i in list(combinations(cards,3)):
        if is_set(i[0],i[1],i[2]):
            count+=1        
    return count

def is_set(a, b, c):
    """Determine if the cards a, b, and c constitute a set.
    Parameters:
    a, b, c (str): string representations of 4-bit integers in base 3.
    For example, "1022", "1122", and "1020" (which is not a set).
    Returns:
    True if a, b, and c form a set, meaning the ith digit of a, b,
    and c are either the same or all different for i=1,2,3,4.
    False if a, b, and c do not form a set.
    """
    for i in range(4):
        if len(set([a[i], b[i], c[i]])) not in [1,3]:
            return False
    return True