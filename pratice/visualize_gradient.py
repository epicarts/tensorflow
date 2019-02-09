import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


'''
참고: https://jed-ai.github.io/py1_gd_animation/
'''
def func_y(x):
    y = x**2 - 4*x + 2
    return y

def gradient_descent(previous_x, learning_rate, epoch):
    x_gd = []
    y_gd = []

    x_gd.append(previous_x)
    y_gd.append(func_y(previous_x))

    # begin the loops to update x and y
    for i in range(epoch):
        current_x = previous_x - learning_rate*(2*previous_x - 4)
        x_gd.append(current_x)
        y_gd.append(func_y(current_x))

        # update previous_x
        previous_x = current_x

    return x_gd, y_gd

x0 = -0.7
learning_rate = 0.15
epoch = 10

# y = x^2 - 4x + 2
x = np.arange(-1, 5, 0.01)
y = func_y(x)

ax = plt.subplot(1, 1, 1)
ax.plot(x, y, lw = 0.9, color = 'k')
ax.set_xlim([min(x), max(x)])
ax.set_ylim([-3, max(y)+1])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

x_gd, y_gd = gradient_descent(x0, learning_rate, epoch)

ax.scatter(x_gd, y_gd, c = 'b')

for i in range(1, epoch+1):
    ax.annotate('', xy=(x_gd[i], y_gd[i]), xytext=(x_gd[i-1], y_gd[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')

plt.show()


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
def func_z(x, y):
    # Calculate values of Z from the created grid
    z = x**2/5. + x*y/50. + y**2/5.

    return z

def gradient_descent(previous_x, previous_y, learning_rate, epoch):
    x_gd = []
    y_gd = []
    z_gd = []

    x_gd.append(previous_x)
    y_gd.append(previous_y)
    z_gd.append(func_z(previous_x, previous_y))

    # begin the loops to update x, y and z
    for i in range(epoch):
        current_x = previous_x - learning_rate*(2*previous_x/5. +
                                               previous_y/50.)
        x_gd.append(current_x)
        current_y = previous_y - learning_rate*(previous_x/50. +
                                                previous_y/5.)
        y_gd.append(current_y)

        z_gd.append(func_z(current_x, current_y))

        # update previous_x and previous_y
        previous_x = current_x
        previous_y = current_y

    return x_gd, y_gd, z_gd

x0 = -2
y0 = 2.5
learning_rate = 1.3
epoch = 10

''' Plot our function '''
a = np.arange(-3, 3, 0.05)
b = np.arange(-3, 3, 0.05)

x, y = np.meshgrid(a, b)
z = func_z(x, y)

fig1, ax1 = plt.subplots()

ax1.contour(x, y, z, levels=np.logspace(-3, 3, 25), cmap='jet')
# Plot target (the minimum of the function)
min_point = np.array([0., 0.])
min_point_ = min_point[:, np.newaxis]
ax1.plot(*min_point_, func_z(*min_point_), 'r*', markersize=10)
ax1.set_xlabel(r'x')
ax1.set_ylabel(r'y')

x_gd, y_gd, z_gd = gradient_descent(x0, y0, learning_rate, epoch)
ax1.plot(x_gd, y_gd, 'bo')

for i in range(1, epoch+1):
    ax1.annotate('', xy=(x_gd[i], y_gd[i]), xytext=(x_gd[i-1], y_gd[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')

plt.show()


'''
3D로 구현
'''
a = np.arange(-3, 3, 0.05)
b = np.arange(-3, 3, 0.05)

x, y = np.meshgrid(a, b)
z = func_z(x, y)#z = x**2/5. + x*y/50. + y**2/5.

fig1 = plt.figure()
ax1 = Axes3D(fig1)
surf = ax1.plot_surface(x, y, z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')

# Plot target (the minimum of the function)
min_point = np.array([0., 0.])
min_point_ = min_point[:, np.newaxis]
ax1.plot(*min_point_, func_z(*min_point_), 'r*', markersize=10)

ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_zlabel(r'$z$')

x_gd, y_gd, z_gd = gradient_descent(x0, y0, learning_rate, epoch)

# Create animation
line, = ax1.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
point, = ax1.plot([], [], [], 'bo')
display_value = ax1.text(2., 2., 27.5, '', transform=ax1.transAxes)

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    display_value.set_text('')

    return line, point, display_value

def animate(i):
    # Animate line
    line.set_data(x_gd[:i], y_gd[:i])
    line.set_3d_properties(z_gd[:i])

    # Animate points
    point.set_data(x_gd[i], y_gd[i])
    point.set_3d_properties(z_gd[i])

    # Animate display value
    display_value.set_text('Min = ' + str(z_gd[i]))

    return line, point, display_value

ax1.legend(loc = 1)

anim = animation.FuncAnimation(fig1, animate, init_func=init,
                               frames=len(x_gd), interval=120,
                               repeat_delay=60, blit=True)

plt.show()
