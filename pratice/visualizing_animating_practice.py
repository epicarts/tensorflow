'''
matplotlib를 사용하여 시각화 연습 및 정리

참고 사이트:
http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/


'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
patch = plt.Circle((5, -5), 0.75, fc='y')

def init():
    patch.center = (5, 5)
    ax.add_patch(patch)
    return patch,

def animate(i):
    x, y = patch.center
    x = 5 + 3 * np.sin(np.radians(i))
    y = 5 + 3 * np.cos(np.radians(i))
    patch.center = (x, y)
    return patch,

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=360,
                               interval=20,
                               blit=True)
anim.save('animation.mp4', fps=30)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
x = [i for i in range(20)]
y = [np.sqrt(i**3) for i in x]

def animate(i):
    global x,y
    r = np.random.normal(0,1,1)
    xar = x
    yar = y + r*y
    #ax1.clear()
    ax1.plot(xar,yar,marker='o')
ani = animation.FuncAnimation(fig,animate,interval=1000)

plt.show()
