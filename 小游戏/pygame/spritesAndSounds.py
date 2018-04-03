import pygame,sys,time,random
from pygame.locals import *

#设置游戏
pygame.init()
mainClock=pygame.time.Clock()

#设置窗口
windowWidth=400
windowHeight=400
windowSurface=pygame.display.set_mode((windowWidth,windowHeight),0,32)
pygame.display.set_caption('Sprites and Sound')#设置标题栏

#设置颜色
black=(0,0,0)

#设置方块
player=pygame.Rect(300,100,40,40)
playerImage=pygame.image.load('../InventWithPython_resources/player.png')
playerStretchedImage=pygame.transform.scale(playerImage,(40,40))#缩小和放大
foodImage=pygame.image.load('../InventWithPython_resources/cherry.png')
foods=[]
foodSize=20
for i in range(20):
    foods.append(pygame.Rect(random.randint(0,windowWidth-foodSize),random.randint(0,windowHeight-foodSize),foodSize,foodSize))
foodCounter=0
newFood=40

#设置键盘
moveLeft=False
moveRight=False
moveUp=False
moveDown=False

moveSpeed=6

#设置音乐
pickUpSound=pygame.mixer.Sound('../InventWithPython_resources/pickup.wav')#播放短音效
pygame.mixer.music.load('../InventWithPython_resources/background.mid')#加载背景音乐
pygame.mixer.music.play(-1,0.0)#循环从头开始播放背景音乐
musicPlaying=True

#游戏运行
while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        if event.type==KEYDOWN:
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
            if event.key==ord('m'):#切换和关闭背景音乐
                if musicPlaying:
                    pygame.mixer.music.stop()
                else:
                    pygame.mixer.music.play(-1,0.0)
                musicPlaying=not musicPlaying
        if event.type==MOUSEBUTTONUP:
            foods.append(pygame.Rect(event.pos[0]-10,event.pos[1]-10,foodSize,foodSize))

    foodCounter+=1
    if foodCounter>=newFood:
        foodCounter=0
        foods.append(pygame.Rect(random.randint(0,windowWidth-foodSize),random.randint(0,windowHeight-foodSize),foodSize,foodSize))
    #绘制背景
    windowSurface.fill(black)
    #移动player
    if moveDown and player.bottom<windowHeight:
        player.top+=moveSpeed
    if moveUp and player.top>0:
        player.top-=moveSpeed
    if moveRight and player.right<windowWidth:
        player.left+=moveSpeed
    if moveLeft and player.left>0:
        player.left-=moveSpeed

    #绘制方块
    windowSurface.blit(playerStretchedImage,player)
    #检查是否吃到食物
    for food in foods[:]:
        if player.colliderect(food):#检查是否和其他方块相碰撞
            foods.remove(food)
            player=pygame.Rect(player.left,player.top,player.width+2,player.height+2)
            playerStretchedImage=pygame.transform.scale(playerImage,(player.width,player.height))
            if musicPlaying:
                pickUpSound.play()#播放背景音乐
    #绘制食物
    for food in foods:
        windowSurface.blit(foodImage,food)#将一个surface对象绘制在另一个surface对象上
    #绘制窗口
    pygame.display.update()#显示set_node()对象上所绘制的任何物体
    mainClock.tick(40)#设置每秒多少帧FPS
