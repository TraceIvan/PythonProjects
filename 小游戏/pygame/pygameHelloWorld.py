import pygame,sys
from pygame.locals import *


pygame.init()#初始化步骤，必须


windowSurface=pygame.display.set_mode((500,400),0,32)#创建一个GUI窗口：500pix宽，400pix高
pygame.display.set_caption('Hello world!')

#设置RGB颜色
BLACK=(0,0,0)
WHITE=(255,255,255)
RED=(255,0,0)
GREEN=(0,255,0)
BLUE=(0,0,255)

basicFont=pygame.font.SysFont(None,48)#None表示使用系统默认字体，48为字体的大小

text=basicFont.render('Hello world!',True,WHITE,BLUE)#需要绘制的文本的字符串；指定是否想要抗锯齿
#rect(left,top,width,height):矩形左上角XY坐标，宽度，高度
textRect=text.get_rect()#返回表示Font对象的大小和位置的Rect对象
textRect.centerx=windowSurface.get_rect().centerx#中央X的坐标
textRect.centery=windowSurface.get_rect().centery

windowSurface.fill(WHITE)#用白色填充整个Surface对象
#绘制多边形：Surface对象、多边形的颜色、需要依次绘制的点的XY坐标元组、（线条宽度）（没有则自动填充）
pygame.draw.polygon(windowSurface,GREEN,((146,0),(291,106),(236,277),(56,277),(0,106)))

pygame.draw.line(windowSurface,BLUE,(60,60),(120,60),4)
pygame.draw.line(windowSurface,BLUE,(120,60),(60,120))
pygame.draw.line(windowSurface,BLUE,(60,120),(120,120),4)

pygame.draw.circle(windowSurface,BLUE,(300,50),20,0)
pygame.draw.ellipse(windowSurface,RED,(300,250,40,80),1)#画椭圆，传递椭圆左上角XY坐标以及椭圆的宽度和高度

pygame.draw.rect(windowSurface,RED,(textRect.left-20,textRect.top-20,textRect.width+40,textRect.height+40))

pixArray=pygame.PixelArray(windowSurface)#颜色元组的列表
pixArray[480][380]=BLACK#把对应坐标的像素改为对应颜色的像素，自动更新到Surface上
del pixArray#从一个Surface对象创建PixelArray对象，将会锁定Surface对象，无法调用blit()函数。用del解锁

windowSurface.blit(text,textRect)#将一个Surface对象的内容绘制到另一个Surface对象上；指定绘制的位置

pygame.display.update()#更新屏幕

while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()