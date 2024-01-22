from math import log as math_log
from math import exp as math_exp
from math import sin as math_sin
from math import cos as math_cos

class Node:
    verbose = False

    def __init__(self, value, parent_nodes=[], operator='input',grad_wrt_parents=[]):
        self.value = value
        self.parent_nodes = parent_nodes
        self.operator = operator
        self.grad_wrt_parents = grad_wrt_parents
        self.child_nodes = []
        self.grad = 0
        if self.verbose == True:
            print('Create a Node:\n\tvalue:{:<15}\n\tparents:{:<30}\n\toperator:{:<50}'.format(
                str(self.value.__round__(3)),
                str([p.value.__round__(3) for p in self.parent_nodes]),
                self.operator.__repr__())
            )

    def __add__(self, other):
        return add(self,other)

    def __radd__(self, other):
        return add(other,self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self,other)

    def __rmul__(self, other):
        return mul(other,self)

def add(node1, node2):
    value = node1.value + node2.value
    parent_nodes = [node1,node2]
    grad_wrt_parents = [1, 1]
    newNode = Node(value, parent_nodes, "add", grad_wrt_parents)
    node1.child_nodes.append(newNode)
    node2.child_nodes.append(newNode)
    return newNode

def sub(node1, node2):
    value = node1.value - node2.value
    parent_nodes = [node1,node2]
    grad_wrt_parents = [1, -1]
    newNode = Node(value, parent_nodes, "sub", grad_wrt_parents)
    node1.child_nodes.append(newNode)
    node2.child_nodes.append(newNode)
    return newNode

def mul(node1, node2):
    value = node1.value* node2.value
    parent_nodes = [node1,node2]
    grad_wrt_parents = [node2.value, node1.value]
    newNode = Node(value, parent_nodes, "mul", grad_wrt_parents)
    node1.child_nodes.append(newNode)
    node2.child_nodes.append(newNode)
    return newNode

def log(node):
    value = math_log(node.value)
    parent_nodes = [node]
    grad_wrt_parents = [1/(node.value*math_log(math_exp(1)))]
    newNode = Node(value, parent_nodes, "log", grad_wrt_parents)
    node.child_nodes.append(newNode)
    return newNode

def sin(node):
    value = math_sin(node.value)
    parent_nodes = [node]
    grad_wrt_parents = [math_cos(node.value)]
    newNode = Node(value, parent_nodes, "sin", grad_wrt_parents)
    node.child_nodes.append(newNode)
    return newNode

def topological_order(rootNode):
    def add_children(node):
        if node not in visited:
            visited.add(node)
            for child in node.child_nodes:
                add_children(child)
            ordering.append(node)
    ordering, visited = [], set()
    add_children(rootNode)
    return reversed(ordering)

def forward(rootNode):
    if rootNode.verbose:
        print("Forward Mode:")
    rootNode.grad = 1
    ordering = topological_order(rootNode)
    for node in ordering:
        partial_derivative = 0
        for i in range(len(node.parent_nodes)):
            dnode_dparent = node.grad_wrt_parents[i]
            dparent_droot = node.parent_nodes[i].grad
            partial_derivative += dnode_dparent * dparent_droot
            node.grad = partial_derivative
        if rootNode.verbose == True:
            print('value: {:<10}|parents: {:<20}|operator: {:<10}|gradient: {:<10}'.format(
                str(node.value.__round__(3)),
                str([p.value.__round__(3) for p in node.parent_nodes]),
                str(node.operator),
                str(node.grad.__round__(3)))
            )

def reverse_topological_order(rootNode):
    def add_parents(node):
        if node not in visited:
            visited.add(node)
            for i in range(len(node.parent_nodes)):
                add_parents(node.parent_nodes[i])
            ordering.append(node)
    ordering, visited = [], set()
    add_parents(rootNode)
    return reversed(ordering)

def backward(rootNode):
    if rootNode.verbose:
        print("Reverse Mode:")
    rootNode.grad = 1
    ordering = reverse_topological_order(rootNode)
    for node in ordering:
        for i in range(len(node.parent_nodes)):
            doutput_dnode = node.grad
            dnode_dparent = node.grad_wrt_parents[i]
            node.parent_nodes[i].grad += doutput_dnode * dnode_dparent
        if rootNode.verbose == True:
            print('value: {:<10}|parents: {:<20}|operator: {:<10}|gradient: {:<10}'.format(
                str(node.value.__round__(3)),
                str([p.value.__round__(3) for p in node.parent_nodes]),
                str(node.operator),
                str(node.grad.__round__(3)))
            )
