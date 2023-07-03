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
In this section, we illustrate the concept of the expression tree, chain rule, and the codes implementing automatic differentiation.
This section is organized as follows: First, the discussion focuses on the operations and the expression tree.
Then, we connect the chain rule and the expression tree with the computational graph.
Last, the purposes of different passes are illustrated.

### Operations, Expression Tree, and the Wrapped Function:
There are two types of operations used: binary operations, such as addition and multiplication, and unary operations, such negation and absolute value.
An equation constructed by variables and operations.
Hence, an algebraic equation can be represented as an expression tree.

**TODO: A Expression Tree**

Each operation has an intermediate outcome.
The outcome directly depends on its intermediate inputs; hence, its intermediate inputs are stated as its parents.
Also, the outcome of an operation directly affects its children.
Besides its parents and children, all the other effects of a node are indirect and pass through its parents and children.

To construct an expression tree, we define a class wrapping *floating-point* data type.
The wrapped data type can be any data type supporting arithmetic operations.
Operator overriding is the key to the automatic generation of the expression tree.

**TODO: A Schematic Diagram of Wrapped Operation**

For each overridden operator, the inputs are passed in.
Then, the overridden operator generates a node to store the inputs as its parents.
Hence, by applying these overridden operators, the expression tree is constructed without any extra effort.

### Chain Rule and the Computation Graph:
The chain rule is a formula that expresses the derivative of the composition of two differentiable function.
It may also be expressed in Leibniz's notation. If a variable $y$ depends on the variable $v$, which itself depends on the variable $x$. In this case, the chain rule is expressed as 
$\dfrac{dy}{dx}=\dfrac{dy}{dv}\dfrac{dv}{dx}$.

Assume that $V$ is the set with all the nodes $v_i$ of an expression tree.
For calculating a derivative $\dfrac{dy}{dx}$ with such an expression tree, we may use

$\dfrac{dy}{dx}=\sum_{\forall v_i\in Parent(y) }\dfrac{dy}{dv_i}\dfrac{dv_i}{dx}$

Since $v_i\in Parent(y)$, the intermediate derivatives $\dfrac{dy}{dv_i}$ can be obtained by inspecting their operations.
Also, $\dfrac{dv_i}{dx}$ can be obtained by recursively solving $\dfrac{dy}{dx}$ with $y=v_i$.

Since the intermediate derivative is used, it is also stored in the nodes.
For an unary operation $y=f(x)$, one intermediate derivative $\dfrac{dy}{dx}$ may be calculated and stored.
Hence, take $y=sin(x)$ as example, the implementation generates a variable $y$ storing informations:
```
```
The member *grad_wrt(x)* is a dictionary storing intermediate gradients with respect to $x$.

In the case of the binary operations $y=f(x_1,x_2)$ two intermediate derivatives $\dfrac{dy}{dx_1}, \dfrac{dy}{dx_2}$ can be obtained.
Hence, for every intermediate variable, we store the intermediate derivatives for the calculation.
Take $y=x_1\times x_2$ as example, the implementation generates a variable $y$ storing informations:
```
```

### Forward Primal Trace:
### Forward Tagent Trace:
### Backward Adjoint Trace:

## Reference:
Papers:
* [Modeling, Inference and Optimization with Composable Differentiable Procedures (Maclaurin, 2016)](https://www.semanticscholar.org/paper/Modeling%2C-Inference-and-Optimization-With-Maclaurin/d5c6ee4468116671dcd811c1518c1dbf54c99e77)
* [Automatic Differentiation in Machine Learning: a Survey (Baydin et al., (2018))](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)

Repositories:
* [HIPS/autograd](https://github.com/HIPS/autograd)
* [karpathy/micrograd](https://github.com/karpathy/micrograd)
* [pytorch/pytorch](https://github.com/pytorch/pytorch)