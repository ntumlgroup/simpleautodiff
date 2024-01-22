# Simple AutoDiff
This repository demonstrates a simple autograd-like implementation of automatic differentiation.

## Project page
Please refer to our [page]() for the document and slides.

## An Example
We have our example referred to in Table 2 in [Automatic Differentiation in Machine Learning: A Survey (Baydin et al., 2018)](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf).
Here, the equation $y=\log(x_1)+x_1x_2+\sin(x_2)$ with $(x_1,x_2)=(2,5)$ is used.
```python
from simpleautodiff import Node
from simpleautodiff import add, sub, mul, log, sin
from simpleautodiff import forward

# create root nodes
x1 = Node(2)
x2 = Node(5)

# create computational graph and evaluate function value
y = sub(add(log(x1), mul(x1, x2)), sin(x2))
print(y.value)

# perform forward-mode autodiff
forward(x1)
print(y.grad)
```
It first creates the computational graph and evaluate the value of $y$ at $(x_1=2,x_2=5)$ at the same time.
Then, based on the function value $y=11.652$, the forward-mode automatic differentiation from $x_1=2$ is performed.