from simpleautodiff import Node, log, sin, backward
Node.verbose = True
x1 = Node(2)
x2 = Node(5)
y = log(x1) + x1*x2 - sin(x2)
backward(y)
