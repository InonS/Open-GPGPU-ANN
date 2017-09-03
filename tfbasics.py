from logging import DEBUG, basicConfig, info
from sys import stdout

import tensorflow as tf

basicConfig(level=DEBUG, stream=stdout)

# 1. Build the computational graph

# scalar constants
x1 = tf.constant(5)
x2 = tf.constant(6)

# arithmetic on constants
arith_result = x1 * x2
info(arith_result)

# tensorflow multiplication operation
mul_result = tf.multiply(x1, x2)
info(mul_result)

# tensor constants
x1 = tf.constant([[5]])
x2 = tf.constant([[6]])

# matrix multiplication
matmul_result = tf.matmul(x1, x2)
info(matmul_result)

# 2. Interactive session: perform operations on computational graph
with tf.Session() as sess:
    arith_fetches = sess.run(arith_result)
    mul_fetches = sess.run(mul_result)
    matmul_fetches = sess.run(matmul_result)

# 3. Output
info(arith_fetches)
info(mul_fetches)
info(matmul_fetches)
