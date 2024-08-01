from math import log as math_log
from math import sin as math_sin
from math import cos as math_cos


class Node:
    verbose = False
    input_count = 0
    intermediate_count = 0

    def __init__(self, value, parent_nodes=[], operator="input",):
        self.value = value
        self.parent_nodes = parent_nodes
        self.child_nodes = []
        self.operator = operator
        self.grad_wrt_parents = []
        self.partial_derivative = 0

        if self.operator == "input":
            Node.input_count += 1
            self.name = "x%d" % (Node.input_count)
        else:
            Node.intermediate_count += 1
            self.name = "v%d" % (Node.intermediate_count)

        if Node.verbose == True:
            print("{:<2} = {:<18} = {:<8}".format(
                self.name,
                self.operator+str([p.name for p in self.parent_nodes]),
                self.value.__round__(3))
            )


def add(node1, node2):
    value = node1.value + node2.value
    parent_nodes = [node1, node2]
    newNode = Node(value, parent_nodes, "add")
    newNode.grad_wrt_parents = [1, 1]
    node1.child_nodes.append(newNode)
    node2.child_nodes.append(newNode)
    return newNode


def sub(node1, node2):
    value = node1.value - node2.value
    parent_nodes = [node1, node2]
    newNode = Node(value, parent_nodes, "sub")
    newNode.grad_wrt_parents = [1, -1]
    node1.child_nodes.append(newNode)
    node2.child_nodes.append(newNode)
    return newNode


def mul(node1, node2):
    value = node1.value * node2.value
    parent_nodes = [node1, node2]
    newNode = Node(value, parent_nodes, "mul")
    newNode.grad_wrt_parents = [node2.value,node1.value]
    node1.child_nodes.append(newNode)
    node2.child_nodes.append(newNode)
    return newNode


def log(node):
    value = math_log(node.value)
    parent_nodes = [node]
    newNode = Node(value, parent_nodes, "log")
    newNode.grad_wrt_parents = [1/(node.value)]
    node.child_nodes.append(newNode)
    return newNode


def sin(node):
    value = math_sin(node.value)
    parent_nodes = [node]
    newNode = Node(value, parent_nodes, "sin")
    newNode.grad_wrt_parents = [math_cos(node.value)]
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
    return list(reversed(ordering))


def forward(rootNode):
    rootNode.partial_derivative = 1
    ordering = topological_order(rootNode)
    for node in ordering[1:]:
        partial_derivative = 0
        for i in range(len(node.parent_nodes)):
            dnode_dparent = node.grad_wrt_parents[i]
            dparent_droot = node.parent_nodes[i].partial_derivative
            partial_derivative += dnode_dparent * dparent_droot
        node.partial_derivative = partial_derivative

        if Node.verbose == True:
            symbol_process = ""
            value_process = ""
            for i in range(len(node.parent_nodes)):
                dnode_dparent = node.grad_wrt_parents[i]
                symbol_process += "(d" + node.name + "/d" + node.parent_nodes[i].name + ")"\
                                  + "(d" + node.parent_nodes[i].name + "/d" + rootNode.name + ") + "
                value_process += "(" + str(dnode_dparent.__round__(3)) + ")(" + \
                    str(node.parent_nodes[i].partial_derivative.__round__(
                        3)) + ") + "
            print('d{:<2}/d{:<2} = {:<45} \n\t= {:<30} = {:<5}'.format(
                node.name,
                rootNode.name,
                symbol_process.strip(" + "),
                value_process.strip(" + "),
                str(node.partial_derivative.__round__(3)))
            )
