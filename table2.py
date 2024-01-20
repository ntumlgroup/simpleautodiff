from simplebigrad import Node, log, sin, forward
Node.verbose = True
x1 = Node(2)
x2 = Node(5)
y = log(x1) + x1*x2 - sin(x2)
forward(x1)