from simplebigrad import WrappedFloat
WrappedFloat.verbose =True
x1 = WrappedFloat(2)
x2 = WrappedFloat(5)
y = WrappedFloat.log(x1) + x1*x2 -WrappedFloat.sin(x2)
y.backward()