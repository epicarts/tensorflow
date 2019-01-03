import numpy as np
import matplotlib.pyplot as plt


X1 = np.arange(0.0, 1.0, 0.01)
X2 = np.arange(0.0, 1.0, 0.01)
w1, w2, theta = 0.5, 0.5, 0.7 #가중치(W)는 다양한 값이 나올 수 있음.(0.5, 0.5, 0.8) 등등
b = -theta# 임계값을 -b로 치환함.
#
'''
0 = b + (w1*x1) + (w2*x2)를 변형하면,
x2 =
'''

fig = plt.figure()#그림 생성
ax = fig.add_subplot(111)


# x 좌표 : X1
# y 좌표 : X2
# w1 *x1 + x2W2 - theta = 0
# 0.5 x1 +
ax.scatter(0, 0, -(w1 * x1 +b) / w2 + b, c='r', marker='x')#계단 함수 적용전
ax.plot_surface(X1,-(w1 * X1 +b) / w2, Z, linewidth=0)
