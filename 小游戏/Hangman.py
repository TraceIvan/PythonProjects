import random
HANGMANPICS=['''

 +---+
 |   |
     |
     |
     |
     |
=========''','''

 +---+
 |   |
 O   |
     |
     |
     |
=========''','''

 +---+
 |   |
 O   |
 |   |
     |
     |
=========''','''

 +---+
 |   |
 O   |
/|   |
     |
     |
=========''','''

 +---+
 |   |
 O   |
/|\  |
     |
     |
=========''','''

 +---+
 |   |
 O   |
/|\  |
/    |
     |
=========''','''

 +---+
 |   |
 O   |
/|\  |
/ \  |
     |
=========''','''

 +---+
 |   |
[O   |
/|\  |
/ \  |
     |
=========''','''

 +---+
 |   |
[O]  |
/|\  |
/ \  |
     |
=========''']

words='ant baboon badger bat bear beaver camel cat clam cobra cougar coyote crow deer dog donkey duck eagle ferret fox ' \
      'frog goat goose hawk lion lizard llama mole monkey moose mule newt otter owl panda parrot pigeon python rabbit ram ' \
      'rat raven rhino salmon seal shark sheep skunk sloth snake spider stork swan tiger toad trout turkey turtle weasel ' \
      'whale wolf wombat zebra'.split()

wordDict={'Colors':'red orange yellow green blue indigo violet white black brown'.split(),
          'Shapes':'square triangle rectangle circle ellipse rhombus trapezoid chevron pentagon hexagon septagon octagon'
              .split(),
          'Fruits':'apple orange lemon pear watermelon grape grapefruit cherry banana cantaloupe mango strawberry tomato'
              .split(),
          'Animals':'bat bear beaver cat cougar crab deer dog donkey duck eagle fish frog goat leech lion lizard monkey '
                    'moose otter owl panda python rabbit rat shark sheep skunk squid tiger turkey turtle weasel whale '
                    'wolf wombat zebra'.split()}
def getRandomWord(wordList):
    wordIndex=random.randint(0,len(wordList)-1)
    return wordList[wordIndex]

def getRandomWord_2(wordDict):
    wordKey=random.choice(list(wordDict.keys()))
    wordIndex=random.randint(0,len(wordDict[wordKey])-1)
    return [wordDict[wordKey][wordIndex],wordKey]

def displayBoard(HANGMANPICS,missedLetters,correctLetters,secretWord):
    print(HANGMANPICS[len(missedLetters)])
    print()

    print('Missed letters:',end=' ')
    for letter in missedLetters:
        print(letter,end=' ')
    print()

    blanks='_'*len(secretWord)

    for i in range(len(secretWord)):
        if secretWord[i] in correctLetters:
            blanks=blanks[:i]+secretWord[i]+blanks[i+1:]

    for letter in blanks:
        print(letter,end=' ')
    print()

def getGuess(alreadyGuessed):
    while True:
        print('Guess a letter：')
        guess=input()
        guess=guess.lower()
        if len(guess)!=1:
            print('Please enter a single letter.')
        elif guess in alreadyGuessed:
            print('You have already guessed that letter. Choose again.')
        elif guess not in 'abcdefghijklmnopqrstuvwxyz':
            print('Please enter a LETTER.')
        else:
            return guess

def playAgain():
    print('Do you want to play again? (yes or no)')
    return input().lower().startswith('y')

print('H A N G M A N')
appearedLetters= ''
correctLetters=''
secretWord,secretKey=getRandomWord_2(wordDict)
gameIsDone=False

while True:
    print('The secret word is in the set: '+secretKey)
    displayBoard(HANGMANPICS, appearedLetters, correctLetters, secretWord)

    guess=getGuess(appearedLetters + correctLetters)

    if guess in secretWord:
        correctLetters=correctLetters+guess

    foundAllLetters=True
    for i in range(len(secretWord)):
        if secretWord[i] not in correctLetters:
            foundAllLetters=False
            break

    if foundAllLetters:
        print('Yes! The secret word is "'+secretWord+'"! You have won!')
        gameIsDone=True
    else:
        appearedLetters= appearedLetters + guess

        if len(appearedLetters)==len(HANGMANPICS)-1:
            displayBoard(HANGMANPICS, appearedLetters, correctLetters, secretWord)
            print('You have run out of guesses!\nAfter ' + str(len(appearedLetters)) + ' appeared guesses and '
                  + str(len(correctLetters)) +' correct guesses, the word was "' + secretWord +'".')
            gameIsDone=True

    if gameIsDone:
        if playAgain():
            appearedLetters= ''
            correctLetters=''
            gameIsDone=False
            secretWord,secretKey=getRandomWord_2(wordDict)
        else:
            break