from math import log as math_log
from math import exp as math_exp
from math import sin as math_sin
from math import cos as math_cos

class WrappedFloat:
    """A wrapped floating-point"""
    __wrapped_type = float
    verbose = False
    def __init__(self, value:__wrapped_type, __parents=[], __op=None) -> None:
        self.value = value
        self.__parents = __parents
        self.__children = []
        self.__op = __op
        self.grad = None
        self.grad_wrt = {}
        if self.verbose == True:
            print('val:{:<10}|par:{:<30}|ops:{:<50}'.format(
                str(self.value.__round__(3)),
                str([p.value.__round__(3) for p in self.__parents]),
                self.__op.__repr__())
                )
    
    def __repr__(self) -> str:
        if self.grad:
            return f'WrappedFloat(value={self.value:.2f}, grad={self.grad:.3f})'
        else:
            return f'WrappedFloat(value={self.value:.2f}, grad=(Unsolved)'

    def add(x1,x2):
        if not WrappedFloat.__instancecheck__(x1):
            x1 = WrappedFloat(x1)
        if not WrappedFloat.__instancecheck__(x2):
            x2 = WrappedFloat(x2)
        fnode = WrappedFloat(x1.value+x2.value, [x1,x2], WrappedFloat.__wrapped_type.__add__)
        fnode.grad_wrt[x1] = 1
        fnode.grad_wrt[x2] = 1
        x1.__children.append(fnode)
        x2.__children.append(fnode)
        return fnode

    def sub(x1,x2):
        if not WrappedFloat.__instancecheck__(x1):
            x1 = WrappedFloat(x1)
        if not WrappedFloat.__instancecheck__(x2):
            x2 = WrappedFloat(x2)
        fnode = WrappedFloat(x1.value-x2.value, [x1,x2], WrappedFloat.__wrapped_type.__sub__)
        fnode.grad_wrt[x1] = 1
        fnode.grad_wrt[x2] = -1
        x1.__children.append(fnode)
        x2.__children.append(fnode)
        return fnode

    def mul(x1,x2):
        if not WrappedFloat.__instancecheck__(x1):
            x1 = WrappedFloat(x1)
        if not WrappedFloat.__instancecheck__(x2):
            x2 = WrappedFloat(x2)
        fnode = WrappedFloat(x1.value * x2.value, [x1, x2], WrappedFloat.__wrapped_type.__mul__)
        fnode.grad_wrt[x1] = x2.value
        fnode.grad_wrt[x2] = x1.value
        x1.__children.append(fnode)
        x2.__children.append(fnode)
        return fnode

    def pow(x1, x2):
        if not WrappedFloat.__instancecheck__(x1):
            x1 = WrappedFloat(x1)
        if not WrappedFloat.__instancecheck__(x2):
            x2 = WrappedFloat(x2)
        fnode = WrappedFloat(x1.value ** x2.value, [x1,x2], WrappedFloat.__wrapped_type.__pow__)
        fnode.grad_wrt[x1] = x2.value * x1.value**(x2.value - 1)
        fnode.grad_wrt[x2] = (x1.value**x2.value)*math_log(x1.value)
        x1.__children.append(fnode)
        x2.__children.append(fnode)
        return fnode

    def div(x1,x2):
        if not WrappedFloat.__instancecheck__(x1):
            x1 = WrappedFloat(x1)
        if not WrappedFloat.__instancecheck__(x2):
            x2 = WrappedFloat(x2)
        fnode = WrappedFloat(x1.value / x2.value, [x1,x2], WrappedFloat.__wrapped_type.__truediv__)
        fnode.grad_wrt[x1] = 1 / x1.value
        fnode.grad_wrt[x2] = -x1.value / x2.value**2
        x1.__children.append(fnode)
        x2.__children.append(fnode)
        return fnode

    def __add__(self,other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self,other):
        return WrappedFloat.sub(self,other)

    def __rsub__(self, other):
        return WrappedFloat.sub(other,self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __truediv__(self, other):
        return WrappedFloat.div(self,other)
    
    def __rtruediv__(self, other):
        return WrappedFloat.div(other,self)
  
    def __pow__(self, other):
        return WrappedFloat.pow(self,other)

    def __rpow__(self,other):
        return WrappedFloat.pow(other,self)

    def __neg__(self):
        return self.mul(-1)

    def log(x ,base=None):
        if not WrappedFloat.__instancecheck__(x):
            x = WrappedFloat(x)
        if base == None:
            base = WrappedFloat(math_exp(1))
        elif not WrappedFloat.__instancecheck__(base):
            base = WrappedFloat(base)
        fnode = WrappedFloat(math_log(x.value,base.value), [x,base], math_log)
        fnode.grad_wrt[x] = 1/(x.value*math_log(base.value))
        fnode.grad_wrt[base] = -math_log(x.value)/(base.value*math_log(base.value**2))
        x.__children.append(fnode)
        base.__children.append(fnode)
        return fnode

    def sin(x):
        if not WrappedFloat.__instancecheck__(x):
            x = WrappedFloat(x)
        fnode = WrappedFloat(math_sin(x.value), [x], math_sin)
        fnode.grad_wrt[x] = math_cos(x.value)
        x.__children.append(fnode)
        return fnode

    def cos(x):
        if not WrappedFloat.__instancecheck__(x):
            x = WrappedFloat(x)
        fnode = WrappedFloat(math_cos(x.value), [x], math_cos)
        fnode.grad_wrt[x] = -math_sin(x.value)
        x.__children.append(fnode)
        return fnode

    def forward(self):
        if self.verbose:
            print("Forward Tangent Trace:")
        def _topological_order():
            def _add_children(node):
                if node not in visited:
                    visited.add(node)
                    for child in node.__children:
                        _add_children(child)
                    ordered.append(node)

            ordered, visited = [], set()
            _add_children(self)
            return ordered

        def _compute_grad_of_children(node):
            for child in node.__children:
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
                    str([p.value.__round__(3) for p in node.__parents]),
                    str(node.grad.__round__(3)))
                    )

    def backward(self):
        if self.verbose:
            print("Reverse Adjoint Trace:")
        def _topological_order():
            def _add_parents(node):
                if node not in visited:
                    visited.add(node)
                    for parent in node.__parents:
                        _add_parents(parent)
                    ordered.append(node)

            ordered, visited = [], set()
            _add_parents(self)
            return ordered

        def _compute_grad_of_parents(node):
            for parent in node.__parents:
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
                    str([p.value.__round__(3) for p in node.__parents]),
                    str(node.grad.__round__(3)))
                    )