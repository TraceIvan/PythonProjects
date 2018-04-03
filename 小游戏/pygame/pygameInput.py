'''大方块在界面弹跳吃掉小方块---碰撞检测--升级版：键盘移动，鼠标创建食物'''
import pygame,sys,random
from pygame.locals import *


#建立游戏
pygame.init()
mainClock=pygame.time.Clock()
#建立窗口
windowWidth=400
windowHeight=400
windowSurface=pygame.display.set_mode((windowWidth,windowHeight),0,32)
pygame.display.set_caption('Input')
#设置方向常量
moveLeft=False
moveRight=False
moveUp=False
moveDown=False
moveSpeed=6
#设置颜色常量
black=(0,0,0)
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
white=(255,255,255)

#设置大方块和食物方块
foodCounter=0
newFood=40
foodSize=20
player=pygame.Rect(300, 100, 50, 50)
foods=[]
for i in range(20):
    foods.append(pygame.Rect(random.randint(0,windowWidth-foodSize),random.randint(0,windowHeight-foodSize),
                             foodSize,foodSize))

#运行游戏循环
while True:
    #检查退出事件
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        if event.type==KEYDOWN:
            #改变键盘变量
            if event.key==K_LEFT or event.key==ord('a'):
                moveRight=False
                moveLeft=True
            if event.key == K_RIGHT or event.key == ord('d'):
                moveRight = True
                moveLeft = False
            if event.key==K_UP or event.key==ord('w'):
                moveDown=False
                moveUp=True
            if event.key==K_DOWN or event.key==ord('s'):
                moveDown=True
                moveUp=False
        if event.type==KEYUP:
            if event.key==K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key==K_LEFT or event.key==ord('a'):
                moveLeft=False
            if event.key == K_RIGHT or event.key == ord('d'):
                moveRight=False
            if event.key == K_UP or event.key == ord('w'):
                moveUp=False
            if event.key==K_DOWN or event.key==ord('s'):
                moveDown=False
            if event.key==ord('x'):
                player.top=random.randint(0,windowHeight-player.height)
                player.left=random.randint(0,windowWidth-player.width)
        if event.type==MOUSEBUTTONUP:
            foods.append(pygame.Rect(event.pos[0],event.pos[1],foodSize,foodSize))

    foodCounter+=1
    if foodCounter>=newFood:
        #添加新的食物方块
        foodCounter=0
        foods.append(pygame.Rect(random.randint(0,windowWidth-foodSize),random.randint(0,windowHeight-foodSize),
                             foodSize,foodSize))
    #绘制黑色背景
    windowSurface.fill(black)

    #运动大方块
    if moveDown and player.bottom<windowHeight:
        player.top+=moveSpeed
    if moveUp and player.top>0:
        player.top-=moveSpeed
    if moveRight and player.right<windowWidth:
        player.left+=moveSpeed
    if moveLeft and player.left>0:
        player.left-=moveSpeed

    #绘制大方块
    pygame.draw.rect(windowSurface, white, player)
    #检查是否碰撞
    for food in foods[:]:
        if player.colliderect(food):
            foods.remove(food)
    #绘制食物方块
    for i in range(len(foods)):
        pygame.draw.rect(windowSurface,green,foods[i])
    #绘制窗口
    pygame.display.update()
    mainClock.tick(40)