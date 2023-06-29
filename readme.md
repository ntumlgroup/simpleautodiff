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

## Reference:
Papers:
* [Modeling, Inference and Optimization with Composable Differentiable Procedures (Maclaurin, 2016)](https://www.semanticscholar.org/paper/Modeling%2C-Inference-and-Optimization-With-Maclaurin/d5c6ee4468116671dcd811c1518c1dbf54c99e77)
* [Automatic Differentiation in Machine Learning: a Survey (Baydin et al., (2018))](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)

Repositories:
* [HIPS/autograd](https://github.com/HIPS/autograd)
* [karpathy/micrograd](https://github.com/karpathy/micrograd)
* [pytorch/pytorch](https://github.com/pytorch/pytorch)