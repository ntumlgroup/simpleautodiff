# Simple AutoDiff
This repository demonstrates a simple implementation of automatic differentiation.

## Project Page
Please refer to our [page]() for the document and slides.

## An Example
The following example generates Table 2 in [Automatic Differentiation in Machine Learning: A Survey (Baydin et al., 2018)](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf), which calculates the partial derivative with respect to the first variable $x_1$.
We consider $y=\log(x_1)+x_1x_2-\sin(x_2)$ with $(x_1,x_2)=(2,5)$.
```python
from simpleautodiff import *

Node.verbose = True

# create root nodes
x1 = Node(2)
x2 = Node(5)

# create computational graph and evaluate function value
y = sub(add(log(x1), mul(x1, x2)), sin(x2))
# perform forward-mode autodiff
forward(x1)
```
It first creates the computational graph and evaluates the value of $y$ at $(x_1=2,x_2=5)$ simultaneously.
```
x1 = input[]           = 2       
x2 = input[]           = 5       
v1 =   log['x1']       = 0.693   
v2 =   mul['x1', 'x2'] = 10      
v3 =   add['v1', 'v2'] = 10.693  
v4 =   sin['x2']       = -0.959  
v5 =   sub['v3', 'v4'] = 11.652 
```
Then, the code performs forward-mode automatic differentiation at $x_1=2$.
```
dx1/dx1 =                                               
        =                                = 1    
dv2/dx1 = (dv2/dx1)(dx1/dx1) + (dv2/dx2)(dx2/dx1)       
        = (5)(1) + (2)(0)                = 5    
dv1/dx1 = (dv1/dx1)(dx1/dx1)                            
        = (0.5)(1)                       = 0.5  
dv3/dx1 = (dv3/dv1)(dv1/dx1) + (dv3/dv2)(dv2/dx1)       
        = (1)(0.5) + (1)(5)              = 5.5  
dv5/dx1 = (dv5/dv3)(dv3/dx1) + (dv5/dv4)(dv4/dx1)       
        = (1)(5.5) + (-1)(0)             = 5.5 
```