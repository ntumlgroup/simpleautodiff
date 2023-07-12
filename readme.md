# Simple Bidirectional Autograd
This repository provides an autograd-like implementation of a bidirectional automatic differentiation.
As the autograd, we use the function wrappers for generating the computation graph.
Also, the tape is used in both the forward- and the reverse-mode automatic differentiation.

## Run the examples:
This examples are taken from Table 2 and Table 3 in [Automatic Differentiation in Machine Learning: a Survey (Baydin et al., (2018))](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf).
In these two examples, the equation $y=\log(x_1)+x_1x_2+\sin(x_2)$ with $(x_1,x_2)=(2,5)$ is used.

In Table 2, it performs the forward primal trace to calculate the value of $y=11.652$ at first.
Then, based on the outcome of forward primal trace, the forward tangent trace (forward-mode automatic differentiation)  from $x_1$ is performed.
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

In Table 3, as in Table 2, it performs the forward primal trace first.
Then, the backward adjoint trace (reverse-mode automatic differentiation) from $y$ is conducted.
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
The key to calculating gradients with the forward tangent trace and the backward adjoint trace is constructing the computational graph.
In the next section, we discuss the concept and the techniques for the implementation of automatic differentiation.

## From Operations to Gradients: A guide to implement Automatic Differentiation
In this section, we illustrate the concept of the expression tree, chain rule, and the codes implementing automatic differentiation.
This section is organized as follows: First, the discussion focuses on the operations and the expression tree.
Then, we connect the chain rule and the expression tree with the computation graph.
Last, the purposes of different passes are illustrated.

### Operations, Expression Tree, and Nodes:
There are two types of operations used: binary operations, such as addition and multiplication, and unary operations, such negation and absolute value.
An equation constructed by variables and operations.
Hence, an algebraic equation can be represented as an expression tree.
For example, the equation $\log(x_1)+x_1x_2-\sin(x_2)$ can be represented by an expression tree.

```mermaid
graph BT;
    x1=2-->v1["v1=log(x1)"];
    x1=2-->v2["v2=mult(x1,x2)"];
    x2=5-->v2["v2=mult(x1,x2)"];
    x2=5-->v3["v3=sin(x2)"];
    v1["v1=log(x1)"]-->v4["v4=add(v1,v2)"];
    v2["v2=mult(x1,x2)"]-->v4["v4=add(v1,v2)"];
    v3["v3=sin(x2)"]-->v5["v5=sub(v4,v3)"];
    v4["v4=add(v1,v2)"]-->v5["v5=sub(v4,v3)"];
    v5["v5=sub(v4,v3)"]-->y=v5
```

The unary operations used are the logarithmic function (log) and the sine function (sin).
The binary operations used are the multiplication (mult), the addition (add), and the substraction (sub).
Each leaf input ($x_1$ and $x_2$) and each execution of operator form a node in the expression tree.
The basic fields of a node include:
* **operation:** This field keeps track of what operation the node performs. If the node is the leaf input, it should be left as None or Null.
* **parents:** To construct a tree, each node needs to keep track of their input nodes as their parents/children. Also, if the node is the leaf input, it should be left as an empty list.
* **value:** Each operation of a node has an intermediate outcome. For instance, the intermediate outcome of the node $v_1=log(x_1)$ is $\log(2)=0.693$. For the leaf nodes, they keep their assigned values in their **value** field.
* **children (optional):** To construct a tree, we may also keep track of the children of nodes rather than their parents. However, keeping parents is beneficial for the Backward trace. On the other hand, a direct list of children can help create a more efficient Forward Tagent Trace. These two properties will be discussed in the section about tracing. 

In the first step of constructing the tree, we need to design the Node.
For brevity, here is a Node class `WrappedFloat` with basic fields that wraps *floating-point* type.
```python
class WrappedFloat:
    __wrapped_type = float
    def __init__(self, value:__wrapped_type, __parents=[], __op=None) -> None:
        self.value = value
        self.__parents = __parents
        self.__op = __op
```
Here is the corresponding implementation of the previous concepts:
* **value:** keeps the *floating-point* value of the intermediate outcome or the assigned value.
* **__parents:** A list contains the parent `WrappedFloat` nodes. It is worth noting that it has to keep the order of the parents since some of the operations, like substraction and division, are order-dependent.
* **__op:** The implementation of the operation field storing the operator or math function of the *floating-point* type.

### Wrapped Operations:
In the last section, we introduced how to construct a minimal node of an expression tree.
In this section, we focus on how to connect them.
One simple technique to construct the expression tree without requiring any extra effort from the user is to use wrapped operations.
For instance, the codes below were originally used to calculate the equation $y=(x_1+x_2)*x_2$ with $(x_1,x_2)=(2,5)$.
```python
>>> x1 = 2
>>> x2 = 5
>>> y = (x1+x2)*x2
```
We may substitute them with WrappedFloat and the customized arithmetic functions of WrappedFloat.
```python
>>> from simplebigrad import WrappedFloat
>>> add = WrappedFloat.add
>>> mul = WrappedFloat.mul
>>> x1 = WrappedFloat(2)
>>> x2 = WrappedFloat(5)
>>> y = mul(add(x1,x2),x2)
```
The `+` and `*` operators are replaced with the addition function `add(x1,x2)` and the multiplication funciton `mul(x1,x2)`.
These customized arithmetic functions are designed to keep track of the relationships between WrappedFloat nodes.
Take the function `add(x1,x2)` as an instance; here is its minimalistic implementation.
```python
class WrappedFloat:
    .
    .
    .
    def add(x1,x2):
        return WrappedFloat(value=x1.value+x2.value, 
                            __parents=[x1,x2],
                            __ops=WrappedFloat.__wrapped_type.__add__)
```
For each customized arithmetic function, the inputs are passed in.
Then, it returns a `WrappedFloat` node initialized with the *floating-point* value of the operation, keeping the information about the operation and the list of parents.
Through the automatic recording of parentage, the expression tree is constructed by easily substituting the original functions with these customized functions.

Also, by overloading [operators](https://docs.python.org/3/library/operator.html), we may use these customized functions more naturally.
```python
class WrappedFloat:
    .
    .
    .
    def __add__(x1,x2):
        return add(x1,x2)
    def __mul__(x1,x2):
        return mul(x1,x2)
```
Since these functions and operators wrapped the original operations, these implementations are called wrapped operations.
```python
>>> from simplebigrad import WrappedFloat
>>> x1 = WrappedFloat(2)
>>> x2 = WrappedFloat(5)
>>> y = (x1+x2)*x2
```

### Chain Rule and the Computation Graph:
In this section, we briefly introduce the chain rule and mention another member worth keeping in each node for making a simple expression tree a computation graph.

The chain rule is a formula that expresses the derivative of the composition of two differentiable function.
It may also be expressed in Leibniz's notation. If a variable $y$ depends on the variable $v$, which itself depends on the variable $x$. In this case, the chain rule is expressed as 
$$\dfrac{dy}{dx}=\dfrac{dy}{dv}\dfrac{dv}{dx}$$.

Assume that $V$ is the set with all the nodes $v_i$ of an expression tree.
For calculating a derivative $\dfrac{dy}{dx}$ with such an expression tree, we may use

$$\dfrac{dy}{dx}=\sum_{\forall v_i\in Parent(y) }\dfrac{dy}{dv_i}\dfrac{dv_i}{dx}$$.

Since $v_i\in Parent(y)$, the intermediate derivatives $\dfrac{dy}{dv_i}$ can be obtained by inspecting their operations.

Also, $\dfrac{dv_i}{dx}$ can be obtained by recursively solving $\dfrac{dy}{dx}$ with $y=v_i$.


https://github.com/Fangop/simplebigrad/blob/2e043bcfb686ad7722ea1bcc299cabf6c758c87d/simplebigrad/simplebigrad.py#L10-L16

Each field member stores different information mentioned above.
Field member `value` is for the numerical outcome of the operation.
Filed members `__parents` and `__op` record the input instances and the operation, these two variables are called recipe in literature.
Member `__children` is initialized as an empty list for the track of the following operations.
In most of the implementations, `__children` is redundant since forward tangent trace is dominated by backward adjoint trace in most of the practical use.
Last, members `grad` and `grad_wrt` record the gradients.
Member `grad_wrt` is a dictionary for the intermediate gradients with respect to the parents of the node.

For an unary operation $y=f(x)$, one intermediate derivative $\dfrac{dy}{dx}$ may be calculated and stored.
Hence, take $y=sin(x)$ as example, the implementation generates a foward node (`fnode`) to store the information of the operation:

https://github.com/Fangop/simplebigrad/blob/2e043bcfb686ad7722ea1bcc299cabf6c758c87d/simplebigrad/simplebigrad.py#L131-L137

We may see that the value of $sin(x)$, the parents list of $[x]$, the operations `math_sin` are the arguements passed for the initialization of the `fnode`.
The intermediate gradient $\dfrac{dsin(x)}{dx}=cos(x)$ is stored in the member field `grad_wrt[x]`.
Then, the parent $x$ also keep track of its child `fnode`.
Eventually, the `fnode` is returned for the following operations.

In the case of the binary operations $y=f(x_1,x_2)$ two intermediate derivatives $\dfrac{dy}{dx_1}, \dfrac{dy}{dx_2}$ can be obtained.
Hence, for every intermediate variable, we store the intermediate derivatives for the calculation.
Take $y=\dfrac{x_1}{x_2}$ as example, the implementation, same, generates a foward node (`fnode`) to store the information of the operation:

https://github.com/Fangop/simplebigrad/blob/2e043bcfb686ad7722ea1bcc299cabf6c758c87d/simplebigrad/simplebigrad.py#L76-L84

Since we implement the division by operator overriding, the parameters `self` and `other` are the $x_1$ and the $x_2$ in the formula, respectively.
Notice that field member `__parents` is a list for recording the order since not all the binary operations are commutative.
Another thing worth noting is that, for binary operations, there are two intermediate gradients with respect to their two parents (inputs).
Hence, $\dfrac{d(x_1/x_2)}{dx_1}=\dfrac{1}{x_2}$ and $\dfrac{d(x_1/x_2)}{dx_2}=-\dfrac{x_1}{(x_2)^2}$ are stored.

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