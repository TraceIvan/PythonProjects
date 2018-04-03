'''设置三种不同颜色的积木四处移动并从墙上反弹'''
import pygame,sys,time
from pygame.locals import *

#建立pygame()
pygame.init()
#建立窗口
windowWidth=400
windowHeight=400
windowSurface=pygame.display.set_mode((windowWidth,windowHeight),0,32)
pygame.display.set_caption('Animation')

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

#设置方块的数据结构
b1={'rect':pygame.Rect(300,80,50,100),'color':red,'dir':upRight}
b2={'rect':pygame.Rect(200,200,20,20),'color':green,'dir':upLeft}
b3={'rect':pygame.Rect(100,150,60,60),'color':blue,'dir':downLeft}
blocks=[b1,b2,b3]
#运行游戏循环
while True:
    #检查退出事件
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
    #在窗口界面绘制黑色背景
    windowSurface.fill(black)

    for b in blocks:
        #移动方块
        if b['dir']==downLeft:
            b['rect'].left-=moveSpeed
            b['rect'].top+=moveSpeed
        if b['dir']==downRight:
            b['rect'].left+=moveSpeed
            b['rect'].top+=moveSpeed
        if b['dir']==upLeft:
            b['rect'].left-=moveSpeed
            b['rect'].top-=moveSpeed
        if b['dir']==upRight:
            b['rect'].left+=moveSpeed
            b['rect'].top-=moveSpeed
        #检查方块是否移到窗口外面，是则修改弹回方向
        if b['rect'].top<0:
            if b['dir']==upLeft:
                b['dir']=downLeft
            if b['dir']==upRight:
                b['dir']=downRight
        if b['rect'].bottom>windowHeight:
            if b['dir']==downLeft:
                b['dir']=upLeft
            if b['dir']==downRight:
                b['dir']=upRight
        if b['rect'].left<0:
            if b['dir']==downLeft:
                b['dir']=downRight
            if b['dir']==upLeft:
                b['dir']=upRight
        if b['rect'].right>windowWidth:
            if b['dir']==downRight:
                b['dir']=downLeft
            if b['dir']==upRight:
                b['dir']=upLeft
        #把方块绘制在界面上
        pygame.draw.rect(windowSurface,b['color'],b['rect'])

    #把窗口绘制在屏幕上
    pygame.display.update()
    time.sleep(0.02)

