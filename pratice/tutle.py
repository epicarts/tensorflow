import turtle
import math
import random

t=turtle
t.up()
t.shape("circle")
t.color("red")



def catch(x,y):
    d = math.sqrt(math.pow(x - target[0],2) + math.pow(y - target[1],2))
    print(d)
    if d < 30:
        t.color('blue')
        t.write("gooooood!!!",False,"center",("",15))
        return True# 만약 성공했으면 리턴
    else:
        t.color('red')
        t.write("fail!!!!!!!",False,"center",("",15))
        return False# 실패했으면 False 리턴

def turnleft():
    player.left(5)

def turnright():
    player.right(5)

def fire():
    x=player.xcor()
    y=player.ycor()
    velocity=80
    angle=player.heading()
    vx=velocity*math.cos(angle*3.14/180.0)
    vy=velocity*math.sin(angle*3.14/180.0)
    while player.ycor()>=0:#포물선 그리면서 날아가는 함수.
        vx=vx
        vy=vy-10
        x=x+vx
        y=y+vy
        player.goto(x, y)
        if catch(x,y) == True:#현재 거북이 위치와 타겟좌표를 계속 비교.
            break

    #제자리에 돌려놓기
    player.home()
    player.goto(-300, 0)


player=turtle.Pen()
player.shape("turtle")
screen=player.getscreen()

screen.onkeypress(turnleft, "Left")
screen.onkeypress(turnright, "Right")
screen.onkeypress(fire, "space")

player.goto(+300, 0)
player.goto(-300, 0)
player.goto(-300, 300)
player.goto(-300, 0)
screen.listen()

target = random.randint(0,30)*10, random.randint(0, 30)*10 #target 위치 값 저장
t.goto(target)

turtle.mainloop()
