import tensorflow as tf

state = tf.Variable(0, name="counter")#0인 변수
one = tf.constant(1) # 명령 생성.
new_value=tf.add(state,one) # 둘이 더한값이 new_value
update = tf.assign(state, new_value)
new_value
update
one
state

init_op = tf.initialize_all_variables()# 변수 초기화 init_op

with tf.Session() as sess:#세션을 열어서 실행함.
    sess.run(init_op)# 초기화 명령 실행
    print(sess.run(state))#state의 초기값을 출력
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
x.initializer.run()
sub = tf.subtract(x, a)
print(sub.eval())
sess.close()

list1=list(map(int,input('입력하세요').strip().split()))
list1
print(list1.sort())
list1.sort()
list1
sorted(list1)
