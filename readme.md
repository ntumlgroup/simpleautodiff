# Simple Bidirectional Autograd
This repository provides an autograd-like implementation of a bidirectional automatic differentiation.
As the autograd, we use the function wrappers for generating the computation graph.
Also, the tape is used in both the forward- and the reverse-mode automatic differentiation.
## Run the examples:
This examples are taken from Table2 and Table3 in [Automatic Differentiation in Machine Learning: a Survey (Baydin et al., (2018))](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)
Table2 is about the forward tangent trace.
```
$ python table2.py
val:2         |par:[]                            |ops:None                                              
val:5         |par:[]                            |ops:None                                              
val:2.718     |par:[]                            |ops:None                                              
val:0.693     |par:[2, 2.718]                    |ops:<built-in function log>                           
val:10        |par:[2, 5]                        |ops:<slot wrapper '__mul__' of 'float' objects>       
val:10.693    |par:[0.693, 10]                   |ops:<slot wrapper '__add__' of 'float' objects>       
val:-0.959    |par:[5]                           |ops:<built-in function sin>                           
val:11.652    |par:[10.693, -0.959]              |ops:<slot wrapper '__sub__' of 'float' objects>       
Forward Tangent Trace:
val:2         |par:[]                            |grad:1                             
val:10        |par:[2, 5]                        |grad:5                             
val:0.693     |par:[2, 2.718]                    |grad:0.5                           
val:10.693    |par:[0.693, 10]                   |grad:5.5                           
val:11.652    |par:[10.693, -0.959]              |grad:5.5 
```
Table3 is about the reverse adjoint trace.
```
$ python table3.py
val:2         |par:[]                            |ops:None                                              
val:5         |par:[]                            |ops:None                                              
val:2.718     |par:[]                            |ops:None                                              
val:0.693     |par:[2, 2.718]                    |ops:<built-in function log>                           
val:10        |par:[2, 5]                        |ops:<slot wrapper '__mul__' of 'float' objects>       
val:10.693    |par:[0.693, 10]                   |ops:<slot wrapper '__add__' of 'float' objects>       
val:-0.959    |par:[5]                           |ops:<built-in function sin>                           
val:11.652    |par:[10.693, -0.959]              |ops:<slot wrapper '__sub__' of 'float' objects>       
Reverse Adjoint Trace:
val:11.652    |par:[10.693, -0.959]              |grad:1                             
val:-0.959    |par:[5]                           |grad:-1                            
val:10.693    |par:[0.693, 10]                   |grad:1                             
val:10        |par:[2, 5]                        |grad:1                             
val:5         |par:[]                            |grad:1.716                         
val:0.693     |par:[2, 2.718]                    |grad:1                             
val:2.718     |par:[]                            |grad:-0.127                        
val:2         |par:[]                            |grad:5.5
```
Both of them are started from a forward primal trace.

## From Operations to Gradients: A guide to implement Automatic Differentiation
In this section, we illustrate the concept of calculus and the codes implementing Automatic Differentiation.
This section is organized as follows: First, we connect the chain rule with the computational graph.
Then, the different passes and traces are introduced.
### Chain Rule and the Graph
The chain rule is a formula that expresses the derivative of the composition of two differentiable function.
It may also be expressed in Leibniz's notation. If a variable $y$ depends on the variable $v$, which itself depends on the variable $x$. In this case, the chain rule is expressed as 

$\frac{dy}{dx} = \frac{dy}{dv} \dot \frac{dv}{dx} $.

There are two types of operations used: binary operations, such as addition and multiplication, and unary operations, such negation and absolute value.
An equation constructed by variables and operations.
An output value $y$ is the outcome of such equations.
Hence, an equation can be represented as a tree.
In the implementation, a node storing intermediate variables $v_i$ is generated after operation to construct the tree.

$Tree\ Graph\ of\ operations$

For calculating a derivative $\frac{dy}{dx_1}$ with $n$ intermediate varaibles $v_i$,

$\frac{dy}{dx_1} = \frac{dy}{dv_1}\dot \frac{dv_1}{dx_1}+ \frac{dy}{dv_2}\dot \frac{dv_2}{dx_1}+\dots+ \frac{dy}{dv_n}\dot \frac{dv_n}{dx_1}$

However, some of the $\frac{dv_1}{dx_1}$ may be zero since $v_i$ and $x_1$ may be independent.
Hence, in every node, we store its parents and childs for excluding redundant values in the summation.

Every intermediate is calculated by an operation. 
For an unary operation $v=f(x)$, one intermediate derivative $\frac{dv}{dx}$ may be calculated and stored.
Hence, take $v=sin(x)$ as example, the implementation generates a variable $v$ storing informations:
```
    node.value=sin(x)
    node.grad_wrt(x) = cos(x)
    node.__parent = [x]
    node.__op = sin(x)
```
The member *grad_wrt(x)* is a dictionary storing intermediate gradients with respect to $x$.

In the case of the binary operations $v=f(x_1,x_2)$ two intermediate derivatives $\frac{dv}{dx_1}, \frac{dv}{dx_2}$ can be obtained.
Hence, for every intermediate variable, we store the intermediate derivatives for the calculation.
Take $v=x_1\times x_2$ as example, the implementation generates a variable $v$ storing informations:
```
    node.value=x1*x2
    node.grad_wrt(x1) = x2
    node.grad_wrt(x2) = x1
    node.__parent = [x1,x2]
    node.__op = *
```
### Forward Primal Trace
### Forward Tagent Trace
### Backward Adjoint Trace

## Reference:
Papers:
* [Modeling, Inference and Optimization with Composable Differentiable Procedures (Maclaurin, 2016)](https://www.semanticscholar.org/paper/Modeling%2C-Inference-and-Optimization-With-Maclaurin/d5c6ee4468116671dcd811c1518c1dbf54c99e77)
* [Automatic Differentiation in Machine Learning: a Survey (Baydin et al., (2018))](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)

Repositories:
* [HIPS/autograd](https://github.com/HIPS/autograd)
* [karpathy/micrograd](https://github.com/karpathy/micrograd)
* [pytorch/pytorch](https://github.com/pytorch/pytorch)