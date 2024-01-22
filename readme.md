# Simple AutoDiff
This repository demonstrates a simple autograd-like implementation of automatic differentiation.

## Project main page
Please refer to our [main page]() for the document and slides.

## An Example
Here, we have two examples taken from Table 2 and Table 3 in [Automatic Differentiation in Machine Learning: a Survey (Baydin et al., (2018))](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf).
In these two examples, the equation $y=\log(x_1)+x_1x_2+\sin(x_2)$ with $(x_1,x_2)=(2,5)$ is used.
```python
from simpleautodiff import Node
from simpleautodiff import add, sub, mul, log, sin
from simpleautodiff import forward

x1 = Node(2)
x2 = Node(5)
y = sub(add(log(x1), mul(x1, x2)), sin(x2))
print(y.value)
forward(x1)
print(y.grad)
```
It first performs the forward mode to evaluate the value of $y$ at $(x_1=2,x_2=5)$.
Then, based on the function value $y=11.652$, the forward-mode automatic differentiation from $x_1=2$ is performed.