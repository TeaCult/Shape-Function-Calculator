###################### WITH RATIONAKL ########################
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Initialize SymPy printing
sp.init_printing()

# Define symbolic variables
x, y = sp.symbols('x y')
L1, L2, L3 = sp.symbols('L1 L2 L3')

def generate_nodes_barycentric(degree):
    nodes = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            k = degree - i - j
            L1_val = sp.Rational(i, degree)
            L2_val = sp.Rational(j, degree)
            L3_val = sp.Rational(k, degree)
            nodes.append((L1_val, L2_val, L3_val))
    return nodes

def generate_monomials(degree):
    monomials = []
    total_degree = degree
    for i in range(total_degree + 1):
        for j in range(total_degree + 1 - i):
            k = total_degree - i - j
            monomials.append((i, j, k))
    return monomials

def construct_shape_functions_linear_system(degree):
    nodes = generate_nodes_barycentric(degree)
    monomials = generate_monomials(degree)
    num_nodes = len(nodes)
    num_monomials = len(monomials)
    shape_functions = []

    if num_nodes != num_monomials:
        raise ValueError("Number of nodes and monomials must be equal for a square system.")

    L1, L2, L3 = sp.symbols('L1 L2 L3')
    monomial_syms = [L1**i * L2**j * L3**k for i, j, k in monomials]

    # Evaluate monomials at nodes to form matrix A
    A = sp.zeros(num_nodes, num_monomials)
    for ni, (L1_n, L2_n, L3_n) in enumerate(nodes):
        for mi, monomial in enumerate(monomial_syms):
            A[ni, mi] = monomial.subs({L1: L1_n, L2: L2_n, L3: L3_n})

    # Solve for each shape function
    for ni in range(num_nodes):
        b = sp.zeros(num_nodes, 1)
        b[ni] = 1  # Shape function is 1 at its own node
        coeffs = A.LUsolve(b)
        # Construct the shape function
        N_i = sum(c * m for c, m in zip(coeffs, monomial_syms))
        shape_functions.append(sp.expand(N_i))

    return shape_functions, nodes

def substitute_barycentric_to_xy(shape_functions):
    x, y = sp.symbols('x y')
    lambda1 = 1 - x - y
    lambda2 = x
    lambda3 = y

    shape_functions_xy = []
    for N_i in shape_functions:
        N_i_xy = N_i.subs({L1: lambda1, L2: lambda2, L3: lambda3})
        shape_functions_xy.append(sp.expand(N_i_xy))
    return shape_functions_xy

# Define degree
degree = 9  # Adjust degree as needed

# Construct shape functions
shape_functions_barycentric, nodes_barycentric = construct_shape_functions_linear_system(degree)

# Substitute barycentric coordinates with x and y
shape_functions = substitute_barycentric_to_xy(shape_functions_barycentric)

# Display the shape functions
for index, sf in enumerate(shape_functions):
    display(sp.Eq(sp.Symbol(f'N_{index+1}'), sf))


def barycentric_to_cartesian(nodes_barycentric):
    # Vertices of the triangle
    V1 = (0, 0)
    V2 = (1, 0)
    V3 = (0, 1)
    vertices = [V1, V2, V3]

    nodes_cartesian = []
    for L1, L2, L3 in nodes_barycentric:
        x_node = L1 * V1[0] + L2 * V2[0] + L3 * V3[0]
        y_node = L1 * V1[1] + L2 * V2[1] + L3 * V3[1]
        nodes_cartesian.append((x_node, y_node))
    return nodes_cartesian


def evaluate_shape_functions(nodes_cartesian, shape_functions):
    num_funcs = len(shape_functions)
    num_nodes = len(nodes_cartesian)
    results = np.zeros((num_funcs, num_nodes))
    x, y = sp.symbols('x y')
    for i, N_i in enumerate(shape_functions):
        for j, (xj, yj) in enumerate(nodes_cartesian):
            N_i_num = N_i.subs({x: xj, y: yj})
            results[i, j] = float(N_i_num.evalf())
    return results

def plot_nodes(nodes_cartesian, degree):
    # Unpack x and y coordinates
    x_coords = [coord[0] for coord in nodes_cartesian]
    y_coords = [coord[1] for coord in nodes_cartesian]

    # Plot the triangle edges
    triangle_x = [0, 1, 0, 0]
    triangle_y = [0, 0, 1, 0]
    plt.plot(triangle_x, triangle_y, 'k-', linewidth=1.5)

    # Plot the nodes
    plt.scatter(x_coords, y_coords, color='red', zorder=5)

    # Annotate nodes with their indices
    for i, (x_coord, y_coord) in enumerate(nodes_cartesian):
        plt.text(x_coord, y_coord, str(i+1), color='blue', fontsize=12, ha='center', va='center')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Nodes of a Degree {degree} Triangular Element')
    plt.axis('equal')
    plt.grid(True)
    plt.show()


# Construct shape functions
shape_functions_barycentric, nodes_barycentric = construct_shape_functions_linear_system(degree)

# Substitute barycentric coordinates with x and y
shape_functions = substitute_barycentric_to_xy(shape_functions_barycentric)

# Convert nodes to Cartesian coordinates
nodes_cartesian = barycentric_to_cartesian(nodes_barycentric)



# Print evaluation matrix
evaluation = evaluate_shape_functions(nodes_cartesian, shape_functions)
np.set_printoptions(precision=6, suppress=True)
print(nodes_cartesian)
print("Shape function evaluations at nodes:")
print(evaluation)


# Check if the evaluation matrix is close to identity matrix
identity_check = np.allclose(evaluation, np.identity(len(shape_functions)), atol=1e-12)
print("Is the evaluation matrix close to the identity matrix?", identity_check)


# Plot the nodes
plot_nodes(nodes_cartesian, degree)



