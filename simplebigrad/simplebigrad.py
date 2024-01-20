from math import log as math_log
from math import exp as math_exp
from math import sin as math_sin
from math import cos as math_cos

class Node:
    """A wrapped floating-point"""
    __wrapped_type = float
    verbose = False

    def __init__(self, value: __wrapped_type, parents=[], operator='input') -> None:
        self.value = value
        self.parents = parents
        self.children = []
        self.operator = operator
        self.grad = None
        self.grad_wrt = {}
        if self.verbose == True:
            print('Create Node:\n\tvalue:{:<15}\n\tparents:{:<30}\n\toperator:{:<50}'.format(
                str(self.value.__round__(3)),
                str([p.value.__round__(3) for p in self.parents]),
                self.operator.__repr__())
            )

    def __repr__(self) -> str:
        if self.grad:
            return f'Node(value={self.value:.2f}, grad={self.grad:.3f})'
        else:
            return f'Node(value={self.value:.2f}, grad=(Unsolved)'


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

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __rpow__(self, other):
        return pow(other, self)

    def __neg__(self):
        return mul(self,-1)

def add(node1, node2):
    if not Node.__instancecheck__(node1):
        node1 = Node(node1)
    if not Node.__instancecheck__(node2):
        node2 = Node(node2)
    childNode = Node(node1.value+node2.value,
                        [node1, node2], 'add')
    childNode.grad_wrt[node1] = 1
    childNode.grad_wrt[node2] = 1
    node1.children.append(childNode)
    node2.children.append(childNode)
    return childNode

def sub(node1, node2):
    if not Node.__instancecheck__(node1):
        node1 = Node(node1)
    if not Node.__instancecheck__(node2):
        node2 = Node(node2)
    childNode = Node(node1.value-node2.value,
                        [node1, node2], 'sub')
    childNode.grad_wrt[node1] = 1
    childNode.grad_wrt[node2] = -1
    node1.children.append(childNode)
    node2.children.append(childNode)
    return childNode

def mul(node1, node2):
    if not Node.__instancecheck__(node1):
        node1 = Node(node1)
    if not Node.__instancecheck__(node2):
        node2 = Node(node2)
    childNode = Node(node1.value * node2.value,
                        [node1, node2], 'mul')
    childNode.grad_wrt[node1] = node2.value
    childNode.grad_wrt[node2] = node1.value
    node1.children.append(childNode)
    node2.children.append(childNode)
    return childNode

def pow(node1, node2):
    if not Node.__instancecheck__(node1):
        node1 = Node(node1)
    if not Node.__instancecheck__(node2):
        node2 = Node(node2)
    childNode = Node(node1.value ** node2.value,
                        [node1, node2], 'pow')
    childNode.grad_wrt[node1] = node2.value * \
        node1.value**(node2.value - 1)
    childNode.grad_wrt[node2] = (
        node1.value**node2.value)*math_log(node1.value)
    node1.children.append(childNode)
    node2.children.append(childNode)
    return childNode

def div(node1, node2):
    if not Node.__instancecheck__(node1):
        node1 = Node(node1)
    if not Node.__instancecheck__(node2):
        node2 = Node(node2)
    childNode = Node(node1.value / node2.value,
                        [node1, node2], 'div')
    childNode.grad_wrt[node1] = 1 / node1.value
    childNode.grad_wrt[node2] = -node1.value / node2.value**2
    node1.children.append(childNode)
    node2.children.append(childNode)
    return childNode


def log(node):
    if not Node.__instancecheck__(node):
        node = Node(node)
    childNode = Node(math_log(node.value),
                        [node], 'log')
    childNode.grad_wrt[node] = 1/(node.value*math_log(math_exp(1)))
    node.children.append(childNode)
    return childNode

def sin(node):
    if not Node.__instancecheck__(node):
        node1 = Node(node)
    childNode = Node(math_sin(node.value), [node], 'sin')
    childNode.grad_wrt[node] = math_cos(node.value)
    node.children.append(childNode)
    return childNode

def cos(node):
    if not Node.__instancecheck__(node):
        node1 = Node(node)
    childNode = Node(math_cos(node1.value), [node], 'cos')
    childNode.grad_wrt[node] = -math_sin(node.value)
    node.children.append(childNode)
    return childNode


def topological_order(rootNode):
    def add_children(node):
        if node not in visited:
            visited.add(node)
            for child in node.children:
                add_children(child)
            ordered.append(node)

    ordered, visited = [], set()
    add_children(rootNode)
    return reversed(ordered)

def forward(rootNode):
    if rootNode.verbose:
        print("Forward Tangent Trace:")

    def _compute_grad_of_children(node):
        for child in node.children:
            Δoutput_Δnode = node.grad
            Δchild_Δnode = child.grad_wrt[node]
            if child.grad == None:
                child.grad = Δoutput_Δnode * Δchild_Δnode
            else:
                child.grad += Δoutput_Δnode * Δchild_Δnode
    rootNode.grad = 1
    ordered = reversed(topological_order(rootNode))
    for node in ordered:
        _compute_grad_of_children(node)
        if rootNode.verbose == True:
            print('value:{:<15}|parents:{:<30}|gradient:{:<30}'.format(
                str(node.value.__round__(3)),
                str([p.value.__round__(3) for p in node.parents]),
                str(node.grad.__round__(3)))
            )

def reverse_topological_order(rootNode):
    def add_parents(node):
        if node not in visited:
            visited.add(node)
            for parent in node.parents:
                add_parents(parent)
            ordered.append(node)

    ordered, visited = [], set()
    add_parents(rootNode)
    return reversed(ordered)


def backward(rootNode):
    if rootNode.verbose:
        print("Reverse Adjoint Trace:")

    def _compute_grad_of_parents(node):
        for parent in node.parents:
            Δoutput_Δnode = node.grad
            Δnode_Δparent = node.grad_wrt[parent]

            if parent.grad == None:
                parent.grad = Δoutput_Δnode * Δnode_Δparent
            else:
                parent.grad += Δoutput_Δnode * Δnode_Δparent

    rootNode.grad = 1
    ordered = reverse_topological_order(rootNode)
    for node in ordered:
        _compute_grad_of_parents(node)
        if rootNode.verbose == True:
            print('value:{:<15}|parents:{:<30}|gradient:{:<30}'.format(
                str(node.value.__round__(3)),
                str([p.value.__round__(3) for p in node.parents]),
                str(node.grad.__round__(3)))
            )