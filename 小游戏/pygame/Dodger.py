'''躲避从顶部落下的敌人'''
import pygame,sys,time,random
from pygame.locals import *

windowWidth=600
windowHeight=600
textColor=(255,255,255)#白色
backroundColor=(0,0,0)#背景颜色

FPS=40
baddieMinSize=10
baddieMaxSize=40
baddieMinSpeed=1
baddieMaxSpeed=8
addNewBaddieRate=6
playerMoveRate=5

def terminate():
    pygame.quit()
    sys.exit()

def waitForPlayerToPressKey():
    while True:
        for event in pygame.event.get():
            if event.type==QUIT:
                terminate()
            if event.type==KEYDOWN:
                if event.key==K_ESCAPE:
                    terminate()
                else:
                    return

def playerHasHitBaddie(playerRect,baddies):
    for b in baddies:
        if playerRect.colliderect(b['rect']):
            return True
    return False

def drawText(text,font,surface,x,y):
    textObj=font.render(text,1,textColor)
    textRect=textObj.get_rect()
    textRect.topleft=(x,y)
    surface.blit(textObj,textRect)

#设置游戏、窗口、鼠标
pygame.init()
mainClock=pygame.time.Clock()
windowSurface=pygame.display.set_mode((windowWidth,windowHeight))#可以设置FULLSCREEN，选择是否全屏
pygame.display.set_caption('Dodger')
pygame.mouse.set_visible(False)

#设置字体
font=pygame.font.SysFont(None,48)#默认字体

#设置声音
gameOverSound=pygame.mixer.Sound('../InventWithPython_resources/gameover.wav')
pygame.mixer.music.load('../InventWithPython_resources/background.mid')

#设置图像
playerImage=pygame.image.load('../InventWithPython_resources/player.png')
playerRect=playerImage.get_rect()
baddieImage=pygame.image.load('../InventWithPython_resources/baddie.png')

#显示Staart
drawText('Dodger',font,windowSurface,(windowWidth/3),(windowHeight/3))
drawText('Press a key to start.',font,windowSurface,int((windowWidth/3))-30,(windowHeight/3)+50)
pygame.display.update()

waitForPlayerToPressKey()

topScore=0
while True:
    baddies=[]
    score=0
    playerRect.topleft=(windowWidth/2,windowHeight-50)
    moveLeft=moveRight=moveUp=moveDown=False
    reverseCheat=slowCheat=False#设置作弊模式
    baddieAddCounter=0
    pygame.mixer.music.play(-1,0.0)
    #游戏开始
    while True:
        score+=1

        for event in pygame.event.get():
            if event.type==QUIT:
                terminate()

            if event.type==KEYDOWN:
                if event.key==ord('z'):
                    reverseCheat=True#敌人方向转置
                if event.key==ord('x'):
                    slowCheat=True
                if event.key==K_LEFT or event.key==ord('a'):
                    moveRight=False
                    moveLeft=True
                if event.key == K_RIGHT or event.key == ord('d'):
                    moveRight = True
                    moveLeft = False
                if event.key == K_UP or event.key == ord('w'):
                    moveDown = False
                    moveUp = True
                if event.key == K_DOWN or event.key == ord('s'):
                    moveDown = True
                    moveUp = False

            if event.type == KEYUP:
                if event.key==ord('z'):
                    reverseCheat=False
                if event.key==ord('x'):
                    slowCheat=False
                if event.key == K_ESCAPE:
                    terminate()

                if event.key == K_LEFT or event.key == ord('a'):
                    moveLeft = False
                if event.key == K_RIGHT or event.key == ord('d'):
                    moveRight = False
                if event.key == K_UP or event.key == ord('w'):
                    moveUp = False
                if event.key == K_DOWN or event.key == ord('s'):
                    moveDown = False
            if event.type==MOUSEMOTION:
                #让player沿着鼠标移动
                playerRect.move_ip(event.pos[0]-playerRect.centerx,event.pos[1]-playerRect.centery)

        #添加baddies
        if not reverseCheat and not slowCheat:
            baddieAddCounter+=1
        if baddieAddCounter==addNewBaddieRate:
            baddieAddCounter=0
            baddieSize=random.randint(baddieMinSize,baddieMaxSize)
            newBaddie={'rect':pygame.Rect(random.randint(0,windowWidth-baddieSize),0-baddieSize,baddieSize,baddieSize),
                       'speed':random.randint(baddieMinSpeed,baddieMaxSpeed),
                       'surface':pygame.transform.scale(baddieImage,(baddieSize,baddieSize))}
            baddies.append(newBaddie)
        #移动player
        if moveLeft and playerRect.left>0:
            playerRect.move_ip(-1*playerMoveRate,0)
        if moveRight and playerRect.right<windowWidth:
            playerRect.move_ip(playerMoveRate,0)
        if moveUp and playerRect.top>0:
            playerRect.move_ip(0,-1*playerMoveRate)
        if moveDown and playerRect.bottom<windowHeight:
            playerRect.move_ip(0,playerMoveRate)

        #移动鼠标来匹配player
        pygame.mouse.set_pos(playerRect.centerx,playerRect.centery)

        #移动baddies
        for b in baddies:
            if not reverseCheat and not slowCheat:
                b['rect'].move_ip(0,b['speed'])
            elif slowCheat:
                b['rect'].move_ip(0,1)
            elif reverseCheat:
                b['rect'].move_ip(0,-5)


        #删除掉到底部的baddies
        for b in baddies[:]:
            if b['rect'].bottom>windowHeight:
                baddies.remove(b)

        #在窗口上绘制背景
        windowSurface.fill(backroundColor)

        #显示分数和最高分数
        drawText('Score: %s'%(score),font,windowSurface,10,0)
        drawText('Top Score: %s'%(topScore),font,windowSurface,10,40)

        #绘制player
        windowSurface.blit(playerImage,playerRect)

        #绘制每一个baddies
        for b in baddies:
            windowSurface.blit(b['surface'],b['rect'])

        pygame.display.update()

        #检查是否有baddies撞到player
        if playerHasHitBaddie(playerRect,baddies):
            if score>topScore:
                topScore=score
            break

        mainClock.tick(FPS)

    #停止游戏并显示game over
    pygame.mixer.music.stop()
    gameOverSound.play()
    drawText('GAME OVER',font,windowSurface,(windowWidth/3),(windowHeight/3))
    drawText('Press a key to play again.',font,windowSurface,(windowWidth/3)-80,(windowHeight/3)+50)
    pygame.display.update()
    waitForPlayerToPressKey()

    gameOverSound.stop()

