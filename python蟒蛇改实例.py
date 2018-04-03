import turtle
#from turtle import *
def drawSnake(rad, angle, len, neckrad):
    for i in range(len):
        turtle.circle(rad, angle)
        turtle.circle(-rad, angle)
    turtle.circle(neckrad,90)
    for i in range(len-1):
        turtle.circle(rad, angle)
        turtle.circle(-rad, angle)
    turtle.circle(neckrad,90)
    for i in range(len):
        turtle.circle(rad, angle)
        turtle.circle(-rad, angle)
    turtle.circle(neckrad,90)
    for i in range(len-1):
        turtle.circle(rad, angle)
        turtle.circle(-rad, angle)
    '''turtle.circle(rad, angle/2)
    turtle.circle(neckrad,180)
    turtle.fd(rad*2/3)'''

def main():
    turtle.setup(1300,1200,0,0)
    pythonsize=30
    turtle.pensize(pythonsize)
    turtle.pencolor("blue")#("#3B9909")(RGB方式)
    turtle.seth(-40)
    drawSnake(40,80,5,pythonsize/2)

main()
