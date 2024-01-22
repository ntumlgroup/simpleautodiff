# Simple AutoDiff
This repository demonstrates a simple autograd-like implementation of automatic differentiation.

## Project main page
Please refer to our [main page]() for the document and slides.

## Run the examples
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
        operator:'input'                                           
Create Node:
        value:5              
        parents:[]                            
        operator:'input'                                           
Create Node:
        value:0.693          
        parents:[2]                           
        operator:'log'                                             
Create Node:
        value:10             
        parents:[2, 5]                        
        operator:'mul'                                             
Create Node:
        value:10.693         
        parents:[0.693, 10]                   
        operator:'add'                                             
Create Node:
        value:-0.959         
        parents:[5]                           
        operator:'sin'                                             
Create Node:
        value:11.652         
        parents:[10.693, -0.959]              
        operator:'sub'                                             
Forward Tangent Trace:
value:2              |parents:[]                            |gradient:1                             
value:10             |parents:[2, 5]                        |gradient:5                             
value:0.693          |parents:[2]                           |gradient:0.5                           
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
        operator:'input'                                           
Create Node:
        value:5              
        parents:[]                            
        operator:'input'                                           
Create Node:
        value:0.693          
        parents:[2]                           
        operator:'log'                                             
Create Node:
        value:10             
        parents:[2, 5]                        
        operator:'mul'                                             
Create Node:
        value:10.693         
        parents:[0.693, 10]                   
        operator:'add'                                             
Create Node:
        value:-0.959         
        parents:[5]                           
        operator:'sin'                                             
Create Node:
        value:11.652         
        parents:[10.693, -0.959]              
        operator:'sub'                                             
Reverse Adjoint Trace:
value:11.652         |parents:[10.693, -0.959]              |gradient:1                             
value:-0.959         |parents:[5]                           |gradient:-1                            
value:10.693         |parents:[0.693, 10]                   |gradient:1                             
value:10             |parents:[2, 5]                        |gradient:1                             
value:5              |parents:[]                            |gradient:1.716                         
value:0.693          |parents:[2]                           |gradient:1                             
value:2              |parents:[]                            |gradient:5.5 
```