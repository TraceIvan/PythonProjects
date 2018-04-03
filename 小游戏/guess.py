#This is a guess the number game
import random

guessesTaken=0

print('Hello! What is your name?')
myName=input()
guess=''
number=random.randint(1,100)
print('Well, '+myName+', I am thinking of a number between 1 and 100.')
while guessesTaken<6:
    print('Take a guess.')
    guess=input()
    guess=int(guess)

    guessesTaken+=1

    if guess<number:
        print('Your guess is too low.')
    elif guess>number:
        print('Your guess is too high.')
    else:
        break
if guess==number:
    guessesTaken=str(guessesTaken)
    print('Good job, '+myName+'! Your guessed my number in '+guessesTaken+' guesses!')
else:
    number=str(number)
    print('Nope. The number I was thinking of was '+number+' .')
