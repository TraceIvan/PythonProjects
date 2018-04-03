#游戏规则：黑方和白方玩家轮流下棋子，在新的棋子和同一个颜色的另一个棋子之间，如果有任何对手的棋子，都将其反转

import random
import sys
def drawBoard(board):
    HLINE='  +---+---+---+---+---+---+---+---+'
    VLINE='  |   |   |   |   |   |   |   |   |'
    print('    1   2   3   4   5   6   7   8')
    print(HLINE)
    for y in range(8):
        #print(VLINE)
        print(y+1,end=' ')
        for x in range(8):
            print('| %s'%(board[x][y]),end=' ')
        print('|')
        #print(VLINE)
        print(HLINE)

def resetBoard(board):#重置游戏板
    for x in range(8):
        for y in range(8):
            board[x][y]=' '

    board[3][3]='X'#设置开始棋子
    board[3][4]='O'
    board[4][3]='O'
    board[4][4]='X'

def getNewBoard():#创建一个新的游戏板数据结构
    board=[]
    for i in range(8):
        board.append([' ']*8)

    return board

def isValidMove(board,tile,xstart,ystart):#判断一次落子是否有效
    if board[xstart][ystart]!=' 'or not isOnBoard(xstart,ystart):
        return False

    board[xstart][ystart]=tile#在游戏板上临时放置玩家的棋子

    if tile=='X':
        otherTile='O'
    else:
        otherTile='X'

    tilesToFFlip=[]#这一步落子可能导致的翻转的所有对手棋子的列表
    for xdirection,ydirection in [[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]:
        x,y=xstart,ystart
        x+=xdirection
        y+=ydirection

        if isOnBoard(x,y) and board[x][y]==otherTile:
            x+=xdirection
            y+=ydirection
            if not isOnBoard(x,y):
                continue
            while board[x][y]==otherTile:
                x+=xdirection
                y+=ydirection
                if not isOnBoard(x,y):
                    break
            if not isOnBoard(x,y):
                continue
            if board[x][y]==tile:#向某一个方向"移动"直到遇到第一个自己的棋子
                while True:
                    x-=xdirection
                    y-=ydirection
                    if x==xstart and y==ystart:
                        break
                    tilesToFFlip.append([x,y])#把可以翻转的棋子加入列表
    board[xstart][ystart]=' '#删去临时放置的棋子
    if len(tilesToFFlip)==0:
        return False
    return tilesToFFlip#如果有可以翻转的棋子，返回翻转的棋子，否则返回False

def isOnBoard(x,y):
    return x>=0 and x<=7 and y>=0 and y<=7

def getBoardWithValidMoves(board,tile):#得到所有有效移动的一个列表
    dupeBoard=getBoardCopy(board)

    for x,y in getValidMoves(dupeBoard,tile):
        dupeBoard[x][y]='.'
    return dupeBoard

def getValidMoves(board,tile):
    validMoves=[]

    for x in range(8):
        for y in range(8):
            if isValidMove(board,tile,x,y)!=False:
                validMoves.append([x,y])
    return validMoves

def getScoreOfBoard(board):
    xscore=0
    oscore=0
    for x in range(8):
        for y in range(8):
            if board[x][y]=='X':
                xscore+=1
            elif board[x][y]=='O':
                oscore+=1
    return {'X':xscore,'O':oscore}

def enterPlayerTile():
    tile=''
    while not (tile=='X' or tile=='O'):
        print('Do you want to be X or O ?')
        tile=input().upper()

    if tile=='X':
        return ['X','O']
    else:
        return ['O','X']

def whoGoesFirst():
    if random.randint(0,1)==0:
        return 'computer'
    else:
        return 'player'

def playAgain():
    print('Do you want to play again?(yes or no)')
    return input().lower().startswith('y')

def makeMove(board,tile,xstart,ystart):
    tilesToFlip=isValidMove(board,tile,xstart,ystart)

    if tilesToFlip==False:
        return False

    board[xstart][ystart]=tile
    for x,y in tilesToFlip:
        board[x][y]=tile
    return True

def getBoardCopy(board):
    dupeBoard=getNewBoard()

    for x in range(8):
        for y in range(8):
            dupeBoard[x][y]=board[x][y]

    return dupeBoard

def isOnCorner(x,y):
    return (x==0 and y==0)or (x==7 and y==0) or (x==0 and y==7) or(x==7 and y==7)

def getPlayerMove(board,playerTile):
    DIGITS1TO8='1 2 3 4 5 6 7 8'.split()
    while True:
        print('Enter your move, or type quit to end the game, or hints to turn off/on hints.')
        move=input().lower()

        if move=='quit':
            return 'quit'
        if move=='hints':
            return 'hints'

        if len(move)==2 and move[0] in DIGITS1TO8 and move[1] in DIGITS1TO8:
            x=int(move[0])-1
            y=int(move[1])-1
            if isValidMove(board,playerTile,x,y)==False:
                continue
            else:
                break
        else:
            print('That is not a valid move. Type the x digit (1-8), then the y digit(1-8).')
            print('For example, 81 will be the top-right corner.')
    return [x,y]

def getComputerMove(board,computerTile):
    possibleMove=getValidMoves(board,computerTile)

    random.shuffle(possibleMove)

    for x,y in possibleMove:
        if isOnCorner(x,y):
            return [x,y]
    bestScore=-1
    for x,y in possibleMove:
        dupeBoard=getBoardCopy(board)
        makeMove(dupeBoard,computerTile,x,y)
        score=getScoreOfBoard(dupeBoard)[computerTile]
        if score>bestScore:
            bestMove=[x,y]
            bestScore=score
    return bestMove

def showPoints(playerTile,computerTile):
    scores=getScoreOfBoard(mainBoard)
    print('You have %s points.  The computer has %s points.'%(scores[playerTile],scores[computerTile]))

def getRandomMove(board,tile):
    return random.choice(getValidMoves(board,tile))

def isOnSide(x,y):
    return x==0 or x==7 or y==0 or y==7

def getCornerSideBestMove(board,tile):
    possibleMoves=getValidMoves(board,tile)

    random.shuffle(possibleMoves)
    for x,y in possibleMoves:
        if isOnCorner(x,y):
            return [x,y]
    for x,y in possibleMoves:
        if isOnSide(x,y):
            return [x,y]

    return getComputerMove(board,tile)

def getSideBestMove(board,tile):
    possibleMoves=getValidMoves(board,tile)
    random.shuffle(possibleMoves)
    for x,y in possibleMoves:
        if isOnSide(x,y):
            return [x,y]
    return getComputerMove(board,tile)

def getWorstMove(board,tile):
    possibleMoves=getValidMoves(board,tile)
    random.shuffle(possibleMoves)
    worstScore=64
    for x,y in possibleMoves:
        dupeBoard=getBoardCopy(board)
        makeMove(dupeBoard,tile,x,y)
        score=getScoreOfBoard(dupeBoard)[tile]
        if score<worstScore:
            worstScore=score
            worstMove=[x,y]
    return worstMove

def getCornerWorstMove(board,tile):
    possibleMoves=getValidMoves(board,tile)
    random.shuffle(possibleMoves)
    for x,y in possibleMoves:
        if isOnCorner(x,y):
            return [x,y]
    return getWorstMove(board,tile)


#--------------------------------
print('Welcome to Reversi!')
xwins=0
owins=0
ties=0
numGames=int(input('Enter number of games to run:'))

for game in range(numGames):
    print('Game #%s:'%(game),end=' ')
    mainBoard=getNewBoard()
    resetBoard(mainBoard)
    if whoGoesFirst()=='player':
        turn='X'
    else:
        turn='O'

    while True:
        if turn=='X':
            otherTile='O'
            x,y=getWorstMove(mainBoard,'X')
            makeMove(mainBoard,'X',x,y)
        else:
            otherTile='X'
            #x,y=getComputerMove(mainBoard,'O')
            x,y=getCornerWorstMove(mainBoard,'O')
            makeMove(mainBoard,'O',x,y)

        if getValidMoves(mainBoard,otherTile)==[]:
            break
        else:
            turn=otherTile

    scores=getScoreOfBoard(mainBoard)
    print('X scored %s points. O scored %s points.'%(scores['X'],scores['O']))

    if scores['X']>scores['O']:
        xwins+=1
    elif scores['X']<scores['O']:
        owins+=1
    else:
        ties+=1

numGames=float(numGames)
xpercent=round(((xwins/numGames)*100),2)
opercent=round(((owins/numGames)*100),2)
tiepercent=round(((ties/numGames)*100),2)
print('X wins %s games (%s%%), O wins %s games (%s%%), ties for %s games (%s%%) of %s games total.'%(xwins,xpercent,owins,
                                                                                                     opercent,ties,
                                                                                                     tiepercent,numGames))