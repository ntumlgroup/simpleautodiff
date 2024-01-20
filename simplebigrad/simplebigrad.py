from math import log as math_log
from math import exp as math_exp
from math import sin as math_sin
from math import cos as math_cos


class Node:
    """A wrapped floating-point"""
    __wrapped_type = float
    verbose = False

    def __init__(self, value: __wrapped_type, parents=[], operator=None) -> None:
        self.value = value
        self.parents = parents
        self.children = []
        self.operator = operator
        self.grad = None
        self.grad_wrt = {}
        if self.verbose == True:
            print('Create Node:\n\tvalue:{:<10}\n\tparents:{:<30}\n\toperator:{:<50}'.format(
                str(self.value.__round__(3)),
                str([p.value.__round__(3) for p in self.parents]),
                self.operator.__repr__())
            )

    def __repr__(self) -> str:
        if self.grad:
            return f'Node(value={self.value:.2f}, grad={self.grad:.3f})'
        else:
            return f'Node(value={self.value:.2f}, grad=(Unsolved)'

    def add(node1, node2):
        if not Node.__instancecheck__(node1):
            node1 = Node(node1)
        if not Node.__instancecheck__(node2):
            node2 = Node(node2)
        childNode = Node(node1.value+node2.value,
                         [node1, node2], Node.__wrapped_type.__add__)
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
                         [node1, node2], Node.__wrapped_type.__sub__)
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
                         [node1, node2], Node.__wrapped_type.__mul__)
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
                         [node1, node2], Node.__wrapped_type.__pow__)
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
                         [node1, node2], Node.__wrapped_type.__truediv__)
        childNode.grad_wrt[node1] = 1 / node1.value
        childNode.grad_wrt[node2] = -node1.value / node2.value**2
        node1.children.append(childNode)
        node2.children.append(childNode)
        return childNode

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return Node.sub(self, other)

    def __rsub__(self, other):
        return Node.sub(other, self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return Node.div(self, other)

    def __rtruediv__(self, other):
        return Node.div(other, self)

    def __pow__(self, other):
        return Node.pow(self, other)

    def __rpow__(self, other):
        return Node.pow(other, self)

    def __neg__(self):
        return self.mul(-1)

    def log(node1, base=None):
        if not Node.__instancecheck__(node1):
            node1 = Node(node1)
        if base == None:
            base = Node(math_exp(1))
        elif not Node.__instancecheck__(base):
            base = Node(base)
        childNode = Node(math_log(node1.value, base.value),
                         [node1, base], math_log)
        childNode.grad_wrt[node1] = 1/(node1.value*math_log(base.value))
        childNode.grad_wrt[base] = - \
            math_log(node1.value)/(base.value*math_log(base.value**2))
        node1.children.append(childNode)
        base.children.append(childNode)
        return childNode

    def sin(node1):
        if not Node.__instancecheck__(node1):
            node1 = Node(node1)
        childNode = Node(math_sin(node1.value), [node1], math_sin)
        childNode.grad_wrt[node1] = math_cos(node1.value)
        node1.children.append(childNode)
        return childNode

    def cos(node1):
        if not Node.__instancecheck__(node1):
            node1 = Node(node1)
        childNode = Node(math_cos(node1.value), [node1], math_cos)
        childNode.grad_wrt[node1] = -math_sin(node1.value)
        node1.children.append(childNode)
        return childNode

    def forward(self):
        if self.verbose:
            print("Forward Tangent Trace:")

        def _topological_order():
            def _add_children(node):
                if node not in visited:
                    visited.add(node)
                    for child in node.children:
                        _add_children(child)
                    ordered.append(node)

            ordered, visited = [], set()
            _add_children(self)
            return ordered

        def _compute_grad_of_children(node):
            for child in node.children:
                Δoutput_Δnode = node.grad
                Δchild_Δnode = child.grad_wrt[node]
                if child.grad == None:
                    child.grad = Δoutput_Δnode * Δchild_Δnode
                else:
                    child.grad += Δoutput_Δnode * Δchild_Δnode
        self.grad = 1
        ordered = reversed(_topological_order())
        for node in ordered:
            _compute_grad_of_children(node)
            if self.verbose == True:
                print('val:{:<10}|par:{:<30}|grad:{:<30}'.format(
                    str(node.value.__round__(3)),
                    str([p.value.__round__(3) for p in node.parents]),
                    str(node.grad.__round__(3)))
                )

    def backward(self):
        if self.verbose:
            print("Reverse Adjoint Trace:")

        def _topological_order():
            def _add_parents(node):
                if node not in visited:
                    visited.add(node)
                    for parent in node.parents:
                        _add_parents(parent)
                    ordered.append(node)

            ordered, visited = [], set()
            _add_parents(self)
            return ordered

        def _compute_grad_of_parents(node):
            for parent in node.parents:
                Δoutput_Δnode = node.grad
                Δnode_Δparent = node.grad_wrt[parent]

                if parent.grad == None:
                    parent.grad = Δoutput_Δnode * Δnode_Δparent
                else:
                    parent.grad += Δoutput_Δnode * Δnode_Δparent

        self.grad = 1
        ordered = reversed(_topological_order())
        for node in ordered:
            _compute_grad_of_parents(node)
            if self.verbose == True:
                print('val:{:<10}|par:{:<30}|grad:{:<30}'.format(
                    str(node.value.__round__(3)),
                    str([p.value.__round__(3) for p in node.parents]),
                    str(node.grad.__round__(3)))
                )
