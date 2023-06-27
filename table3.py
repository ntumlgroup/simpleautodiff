from simplebigrad import Real
Real.verbose =True
x1 = Real(2)
x2 = Real(5)
y = Real.log(x1) + x1*x2 -Real.sin(x2)
y.backward()