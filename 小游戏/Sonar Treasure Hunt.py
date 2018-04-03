# Sonar
import random
import sys

def drawBoard(board):#绘制游游戏板
    hline='    '
    for i in range(1,6):#表示坐标轴上的10，20，30，40，50
        hline+=(' '*9)+str(i)

    print(hline)#打印X轴坐标
    print('   '+('0123456789'*6))
    print()

    for i in range(15):#打印出海洋的每一行，并且在屏幕的两边都打印出y轴坐标
        if i <10:#如果是1位数，有前导空格
            extraSpace=' '
        else:
            extraSpace=''
        print('%s%s %s %s'%(extraSpace,i,getRow(board,i),i))

    print()
    print('   '+('0123456789'*6))#在底部打印X轴坐标
    print(hline)

def getRow(board,row):#获取海洋中一行的状态
    boardRow=''
    for i in range(60):
        boardRow+=board[i][row]
    return boardRow

def getNewBoard():#创建一个新的游戏板
    board=[]
    for x in range(60):
        board.append([])
        for y in range(15):
            if random.randint(0,1)==0:
                board[x].append('~')
            else:
                board[x].append('`')
    return board

def getRandomChests(numChests):#创建随机的藏宝箱
    chests=[]
    for i in range(numChests):
        chests.append([random.randint(0,59),random.randint(0,14)])
    return chests

def isValidMove(x,y):#判断一次移动是否有效
    return x>=0 and x<=59 and y>=0 and y<=14

def makeMove(board,chests,x,y):#在游戏板上进行一次移动
    if not isValidMove(x,y):
        return False

    smallestDistance=100
    for cx,cy in chests:
        if abs(cx-x)>abs(cy-y):
            distance=abs(cx-x)
        else:
            distance=abs(cy-y)

        if distance<smallestDistance:
            smallestDistance=distance

    if smallestDistance==0:
        chests.remove([x,y])
        return 'You have found a sunken treasure chest!'
    else:
        if smallestDistance<10:
            board[x][y]=str(smallestDistance)
            return 'Treasure detected at a distance of %s from the sonar device.'%(smallestDistance)
        else:
            board[x][y]='0'
            return 'Sonar did not detect anything. All treasure chests out of range.'

def enterPlayerMove():#获取玩家的移动
    print('Where do you want to drop the next sonar device?(0-59 0-14)(or type quit)')
    while  True:
        move=input()
        if move.lower()=='quit':
            print('Thanks for playing!')
            sys.exit()

        move=move.split()
        if len(move)==2 and move[0].isdigit() and move[1].isdigit() and isValidMove(int(move[0]),int(move[1])):
            return [int(move[0]),int(move[1])]
        print('Enter a number from 0 to 59, a space, then a number from 0 to 14.')

def playAgain():#询问是否再玩一局
    print('Do you want to play again?(yes or no)')
    return input().lower().startswith('y')

def showInstructions():#打印游戏说明
    print('''Instructions:
    You are the captain of the Simon, a treasure-hunting ship. Your current mission
    is to find the three sunken treasure chests that are lurking in the part of the
    ocean you are in and collect them.
    
    To play, enter the coordinates of the point in the ocean you wish to drop a
    sonar device. The sonar can find out how far away the closest chest is to it.
    For example, the d below marks where the device was dropped, and the 2's
    represent distances of 2 away from the device. The 4's represent 
    distances of 4 away from the device.
    
        444444444
        4       4
        4 22222 4
        4 2   2 4
        4 2 d 2 4
        4 2   2 4
        4 22222 4
        4       4
        444444444
    Press enter to continue...''')
    input()

    print('''For example, here is a treasure chest (the c) located a distance of 2 away
    from the sonar device (the d):
    
        22222
        c   2
        2 d 2
        2   2
        22222
        
    The point where the device was dropped will be marked with a 2.
    
    The treasure chests don't move around. Sonar devices can detect treasure
    chests up to a distances of 9. If all chests are out of range, the point
    will be marked with 0.
    
    If a device is directly dropped on a treasure chest, you have discovered
    the location of the chest, and it will be collected. The sonar device will
    remain there.
    
    When you collect a chest, all sonar devices will update to locate the next 
    closest sunken treasure chest.
    Press enter to continue...''')
    input()
    print()
#游戏开始
print('S O N A R !')
print()
print('Would you like to view the instructions?(yes/no)')
if input().lower().startswith('y'):
    showInstructions()

while True:
    sonarDevices=16
    theBoard=getNewBoard()
    theChests=getRandomChests(3)
    drawBoard(theBoard)
    previousMoves=[]

    while sonarDevices>0:#为玩家显示游戏的状态
        if sonarDevices>1:extraSsonar='s'
        else:extraSsonar=''
        if len(theChests)>1:extraSchest='s'
        else:extraSchest=''
        print('You have %s sonar device%s left. %s treasure chest%s remaining.'%
              (sonarDevices,extraSsonar,len(theChests),extraSchest))

        x,y=enterPlayerMove()#获取玩家的移动
        previousMoves.append([x,y])

        moveResult=makeMove(theBoard,theChests,x,y)
        if moveResult==False:
            continue
        else:#找到一个沉没的藏宝箱
            if moveResult=='You have found a sunken treasure chest!':
                for x,y in previousMoves:
                    makeMove(theBoard,theChests,x,y)
            drawBoard(theBoard)
            print(moveResult)

        if len(theChests)==0:#判断玩家是否赢了
            print('You have found all the sunken treasure chests!'
                  'Congratulations and good game!')
            break

        sonarDevices-=1

    if sonarDevices==0:#判断玩家是否输了
        print('We\'ve run of sonar devices! Now we have to turn the ship around and head')
        print('for home with treasure chests still out there! Game over.')
        print('    The remaining chests were here:')
        for x,y in theChests:
            print('    %s, %s'%(x,y))

        if not playAgain():
            sys.exit()


