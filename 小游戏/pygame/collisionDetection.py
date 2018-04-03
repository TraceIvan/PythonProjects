'''大方块在界面弹跳吃掉小方块---碰撞检测'''
import pygame,sys,random
from pygame.locals import *

def doRectsOverlap(rect1,rect2):
    for a,b in [(rect1,rect2),(rect2,rect1)]:
        #检测a的角落是否在b内部
        if ((isPointInsideRect(a.left,a.top,b)) or (isPointInsideRect(a.left,a.bottom,b)) or
            (isPointInsideRect(a.right,a.top,b)) or (isPointInsideRect(a.right,a.bottom,b))):
            return True
    return False

def isPointInsideRect(x,y,rect):
    if(x>rect.left) and (x<rect.right) and (y>rect.top) and (y<rect.bottom):
        return True
    else:
        return False

#建立游戏
pygame.init()
mainClock=pygame.time.Clock()
#建立窗口
windowWidth=400
windowHeight=400
windowSurface=pygame.display.set_mode((windowWidth,windowHeight),0,32)
pygame.display.set_caption('Collision Detection')
#设置方向常量
downLeft=1
downRight=3
upLeft=7
upRight=9
moveSpeed=4
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
bouncer={'rect':pygame.Rect(300,100,50,50),'dir':upLeft}
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

    foodCounter+=1
    if foodCounter>=newFood:
        #添加新的食物方块
        foodCounter=0
        foods.append(pygame.Rect(random.randint(0,windowWidth-foodSize),random.randint(0,windowHeight-foodSize),
                             foodSize,foodSize))
    #绘制黑色背景
    windowSurface.fill(black)

    #运动大方块
    if bouncer['dir'] == downLeft:
        bouncer['rect'].left -= moveSpeed
        bouncer['rect'].top += moveSpeed
    if bouncer['dir'] == downRight:
        bouncer['rect'].left += moveSpeed
        bouncer['rect'].top += moveSpeed
    if bouncer['dir'] == upLeft:
        bouncer['rect'].left -= moveSpeed
        bouncer['rect'].top -= moveSpeed
    if bouncer['dir'] == upRight:
        bouncer['rect'].left += moveSpeed
        bouncer['rect'].top -= moveSpeed

    #检查是否出界
    if bouncer['rect'].top < 0:
        if bouncer['dir'] == upLeft:
            bouncer['dir'] = downLeft
        if bouncer['dir'] == upRight:
            bouncer['dir'] = downRight
    if bouncer['rect'].bottom > windowHeight:
        if bouncer['dir'] == downLeft:
            bouncer['dir'] = upLeft
        if bouncer['dir'] == downRight:
            bouncer['dir'] = upRight
    if bouncer['rect'].left < 0:
        if bouncer['dir'] == downLeft:
            bouncer['dir'] = downRight
        if bouncer['dir'] == upLeft:
            bouncer['dir'] = upRight
    if bouncer['rect'].right > windowWidth:
        if bouncer['dir'] == downRight:
            bouncer['dir'] = downLeft
        if bouncer['dir'] == upRight:
            bouncer['dir'] = upLeft
    #绘制大方块
    pygame.draw.rect(windowSurface,white,bouncer['rect'])
    #检查是否碰撞
    for food in foods[:]:
        if doRectsOverlap(bouncer['rect'],food):
            foods.remove(food)
    #绘制食物方块
    for i in range(len(foods)):
        pygame.draw.rect(windowSurface,green,foods[i])
    #绘制窗口
    pygame.display.update()
    mainClock.tick(40)