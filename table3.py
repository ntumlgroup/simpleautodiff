from simplebigrad import Node
Node.verbose =True
x1 = Node(2)
x2 = Node(5)
y = Node.log(x1) + x1*x2 -Node.sin(x2)
y.backward()