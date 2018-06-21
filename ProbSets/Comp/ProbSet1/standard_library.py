import box
import time
import sys
import random
from itertools import combinations

def power(A):
    """
    but no emptyset
    """
    powerset = []
    for i in range(1,len(A)+1):
        subset = combinations(A,i)
        powerset += subset
    for j in range(len(powerset)):
        powerset[j] = set(powerset[j])
    return powerset

def prompt(numbers, startTime, timeLimit, roll):
    secondsRemaining = round(startTime + timeLimit - time.time(),2)
    if secondsRemaining <= 0:
        return "out of time"
    print("Seconds left: " + str(secondsRemaining))
    entry = input("Numbers to eliminate: ")

    # NOTE:
    # I decided to do the entry validation using my own methodology, since it's been a while since I last used Python
    # and I figured I needed more practice. Below (commented out) I have implemented the same thing using the box module.

    numsToEliminate = []
    shouldBeSpace = False

    numbersAsStrings = []
    for j in numbers:
        numbersAsStrings.append(str(j))

    numsToEliminateAsStrings = []

    for i in entry:
        if (i != " " and shouldBeSpace) or (i not in numbersAsStrings and not shouldBeSpace):
            print("Invalid input")
            return "invalid"
        elif not shouldBeSpace:
            for k in numsToEliminate:
                numsToEliminateAsStrings.append(str(k))
            if i in numsToEliminateAsStrings:
                print("Invalid input")
                return "invalid"
            else:
                numsToEliminate.append(int(i))

        shouldBeSpace = not shouldBeSpace

    if sum(numsToEliminate) != roll:
        print("Invalid input")
        return "invalid"

    return numsToEliminate

    # NOTE:
    # I can instead do the following, using the box module:
    # box.parse_input(entry, list(numbers))

    # continue...



def main():
    playerName = sys.argv[1]
    timeLimit = float(sys.argv[2])

    numbers = set(range(1,10))

    numsToEliminate = []

    roll = 0

    elapsedTime = 0.0

    startTime = time.time()

    lose = False

    while (sum(numbers) > 0) and (not lose):
        elapsedTime = time.time() - startTime

        possibleChoices = power(numbers)
        for i in range(len(possibleChoices)):
            possibleChoices[i] = sum(possibleChoices[i])

        if elapsedTime > timeLimit:
            print("Game over!")
            lose = True
            break

        if sum(numbers) > 6:
            roll = random.choice(list(range(1,7))) + random.choice(list(range(1,7)))            
        else:
            roll = random.choice(list(range(1,7)))

        print("Numbers left: " + str(numbers))
        print("Roll: " + str(roll))

        if roll not in possibleChoices:
            print("Game over!\n")
            lose = True
            break

        invalid = True
        while invalid:
            numsToEliminate = prompt(numbers, startTime, timeLimit, roll)
            print("")
            if numsToEliminate == "invalid":
                invalid = True
            elif numsToEliminate == "out of time":
                print("Game over!\n")
                lose = True
                invalid = False
            else:
                invalid = False
        
        if not lose:
            for i in numsToEliminate:
                numbers.remove(i)
            
    
    elapsedTime = time.time() - startTime

    if sum(numbers) == 0:
        print("Score for player " + playerName + ": " + str(sum(numbers)) + " points")
        print("Time played: " + str(round(elapsedTime,2)) + " seconds")
        print("Congratulations!! You shut the box! :)")

    if lose:
        print("Score for player " + playerName + ": " + str(sum(numbers)) + " points")
        print("Time played: " + str(round(elapsedTime,2)) + " seconds")
        print("Better luck next time! >:)")


if __name__ == "__main__" and len(sys.argv) == 3:
    main()

else:
    print("Exactly two extra command line argument is required")
    print("System Arguments:", sys.argv2)