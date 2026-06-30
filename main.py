from simpleautodiff import *

def main():
    Node.verbose = True

    # create root nodes
    x1 = Node(2)
    x2 = Node(5)

    # create computational graph and evaluate function value
    y = sub(add(log(x1), mul(x1, x2)), sin(x2))
    # perform forward-mode autodiff
    print("\n--- Forward mode (root = x1) ---")
    forward(x1)
    print("\n--- Forward mode (root = x2) ---")
    forward(x2)

    # perform reverse-mode autodiff
    print("\n--- Reverse mode (output = y) ---")
    backward(y)
    print("\ndy/dx1 =", x1.partial_derivative.__round__(3))
    print("dy/dx2 =", x2.partial_derivative.__round__(3))

if __name__ == "__main__":
    main()
