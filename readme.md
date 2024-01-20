# Simple Bidirectional Autograd
This repository demonstrates a simple autograd-like implementation of bidirectional automatic differentiation.
As the autograd, we use the function wrappers for generating the computation graph.
Also, the tape is used in both the forward mode and the reverse mode.

We started this project for people who know about automatic differentiation and want to learn how to implement it.
Besides codes, we share tutorial slides and documents.

## Run the examples:
Here, we have two examples taken from Table 2 and Table 3 in [Automatic Differentiation in Machine Learning: a Survey (Baydin et al., (2018))](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf).
In these two examples, the equation $y=\log(x_1)+x_1x_2+\sin(x_2)$ with $(x_1,x_2)=(2,5)$ is used.

In Table 2, it first performs the forward model to evaluate the value of $y=11.652$ at $(x_1=2,x_2=5)$.
Then, based on the function value, the forward-mode automatic differentiation from $x_1$ is performed.

We can run them with scripts `table2.py` and `table3.py`.
```
$ python table2.py
Create Node:
        value:2              
        parents:[]                            
        operator:None                                              
Create Node:
        value:5              
        parents:[]                            
        operator:None                                              
Create Node:
        value:2.718          
        parents:[]                            
        operator:None                                              
Create Node:
        value:0.693          
        parents:[2, 2.718]                    
        operator:<built-in function log>                           
Create Node:
        value:10             
        parents:[2, 5]                        
        operator:<slot wrapper '__mul__' of 'float' objects>       
Create Node:
        value:10.693         
        parents:[0.693, 10]                   
        operator:<slot wrapper '__add__' of 'float' objects>       
Create Node:
        value:-0.959         
        parents:[5]                           
        operator:<built-in function sin>                           
Create Node:
        value:11.652         
        parents:[10.693, -0.959]              
        operator:<slot wrapper '__sub__' of 'float' objects>       
Forward Tangent Trace:
value:2              |parents:[]                            |gradient:1                             
value:10             |parents:[2, 5]                        |gradient:5                             
value:0.693          |parents:[2, 2.718]                    |gradient:0.5                           
value:10.693         |parents:[0.693, 10]                   |gradient:5.5                           
value:11.652         |parents:[10.693, -0.959]              |gradient:5.5
```
In Table 3, as in Table 2, it first evaluates function value.
Then, the reverse-mode automatic differentiation from $y$ is conducted.
```
$ python table3.py
Create Node:
        value:2              
        parents:[]                            
        operator:None                                              
Create Node:
        value:5              
        parents:[]                            
        operator:None                                              
Create Node:
        value:2.718          
        parents:[]                            
        operator:None                                              
Create Node:
        value:0.693          
        parents:[2, 2.718]                    
        operator:<built-in function log>                           
Create Node:
        value:10             
        parents:[2, 5]                        
        operator:<slot wrapper '__mul__' of 'float' objects>       
Create Node:
        value:10.693         
        parents:[0.693, 10]                   
        operator:<slot wrapper '__add__' of 'float' objects>       
Create Node:
        value:-0.959         
        parents:[5]                           
        operator:<built-in function sin>                           
Create Node:
        value:11.652         
        parents:[10.693, -0.959]              
        operator:<slot wrapper '__sub__' of 'float' objects>       
Reverse Adjoint Trace:
value:11.652         |parents:[10.693, -0.959]              |gradient:1                             
value:-0.959         |parents:[5]                           |gradient:-1                            
value:10.693         |parents:[0.693, 10]                   |gradient:1                             
value:10             |parents:[2, 5]                        |gradient:1                             
value:5              |parents:[]                            |gradient:1.716                         
value:0.693          |parents:[2, 2.718]                    |gradient:1                             
value:2.718          |parents:[]                            |gradient:-0.127                        
value:2              |parents:[]                            |gradient:5.5   
```
Everyone can refer to our materials for plain illustrations of the implementation.

## Download:

## Document and Slides
The document and slides can be found at [].
These materials introduce basic concepts and focus on implementation.
We first explain why we need a computational graph to apply the chain rule.
Then, we illustrate how to design nodes and use them to construct the graph.
With the graph, the implementation of the forward mode is discussed.

## Lectures:
The video of lecture can be found at [].
We use these materials in the course `Optimization Methods for Deep Learning`.

## Reference:
Papers:
* [Modeling, Inference and Optimization with Composable Differentiable Procedures (Maclaurin, 2016)](https://www.semanticscholar.org/paper/Modeling%2C-Inference-and-Optimization-With-Maclaurin/d5c6ee4468116671dcd811c1518c1dbf54c99e77)
* [Automatic Differentiation in Machine Learning: a Survey (Baydin et al., (2018))](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)

Repositories:
* [HIPS/autograd](https://github.com/HIPS/autograd)
* [karpathy/micrograd](https://github.com/karpathy/micrograd)
* [pytorch/pytorch](https://github.com/pytorch/pytorch)