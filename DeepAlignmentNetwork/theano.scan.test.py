import numpy as np
import theano
import theano.tensor as T
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
n = theano.shared(floatX(0))
k = T.iscalar("k")

# 这里 lambda 的返回值是一个 dict，因此这个值会被传入 updates 中
# _, updates = theano.scan(fn = lambda n: {n:n+1},
#                          non_sequences = n,
#                          n_steps = k)

# counter = theano.function(inputs = [k],
#                           outputs = [],
#                           updates = updates)

# print (n.get_value())
# counter(10)
# print (n.get_value())
# counter(10)
# print (n.get_value())


x = T.vector()

results, _ = theano.scan(fn = lambda t: t * 2, sequences = x)
x_double_scan = theano.function([x], results)

print(x_double_scan(range(10)))


x = T.ivector('x')
y = theano.shared(np.array([2, 2]), name='y_shared')
inc = theano.shared(np.array([1, 1], dtype=np.int32))
output = x + y
# theano.function有2个输出，先通过output输出一下，更新updates里面的shared parameter
test = theano.function([x], output, updates=[(y, y + inc)])  # updates是一个list量,他的元素是(shared_variable,new expression)对，有点像字典，每次对keyword对应的值更新

input_x = np.array([0, 0])                     # input_x ==(0,0)
print('Output value is : ', test(input_x))     #  output (2,2) = (0,0) + y (2,2)， update: y (2,2) --> y (3,3)
print('The shared variable : ', y.get_value()) # The shared variable :  [3 3]
input_x = np.array([0, 0])                     # input_x ==(0,0)
print('Output value is : ', test(input_x))     # output (3,3) = (0,0) + y (3,3)， update: y (3,3) --> y (4,4)
print('The shared variable : ', y.get_value()) # y (4,4)
