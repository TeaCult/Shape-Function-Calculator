# Shape Function Calculator
# https://github.com/TeaCult/Shape-Function-Calculator
# Copyright (c) 2023 Gediz GÜRSU
# Released under the MIT License

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Initialize SymPy printing
sp.init_printing()

# Define symbolic variables
x, y, z = sp.symbols('x y z')
L1, L2, L3, L4 = sp.symbols('L1 L2 L3 L4')

def generate_nodes_barycentric_tetrahedral(degree):
    nodes = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            for k in range(degree + 1 - i - j):
                l = degree - i - j - k
                L1_val = sp.Rational(i, degree)
                L2_val = sp.Rational(j, degree)
                L3_val = sp.Rational(k, degree)
                L4_val = sp.Rational(l, degree)
                nodes.append((L1_val, L2_val, L3_val, L4_val))
    return nodes

def generate_monomials_tetrahedral(degree):
    monomials = []
    total_degree = degree
    for i in range(total_degree + 1):
        for j in range(total_degree + 1 - i):
            for k in range(total_degree + 1 - i - j):
                l = total_degree - i - j - k
                monomials.append((i, j, k, l))
    return monomials

def construct_shape_functions_linear_system_tetrahedral(degree):
    nodes = generate_nodes_barycentric_tetrahedral(degree)
    monomials = generate_monomials_tetrahedral(degree)
    num_nodes = len(nodes)
    num_monomials = len(monomials)
    shape_functions = []

    if num_nodes != num_monomials:
        raise ValueError("Number of nodes and monomials must be equal for a square system.")

    # Monomial symbols
    L1, L2, L3, L4 = sp.symbols('L1 L2 L3 L4')
    monomial_syms = [L1**i * L2**j * L3**k * L4**l for i, j, k, l in monomials]

    # Evaluate monomials at nodes to form matrix A
    A = sp.zeros(num_nodes, num_monomials)
    for ni, (L1_n, L2_n, L3_n, L4_n) in enumerate(nodes):
        for mi, monomial in enumerate(monomial_syms):
            A[ni, mi] = monomial.subs({L1: L1_n, L2: L2_n, L3: L3_n, L4: L4_n})

    # Solve for each shape function
    for ni in range(num_nodes):
        b = sp.zeros(num_nodes, 1)
        b[ni] = 1  # Shape function is 1 at its own node
        coeffs = A.LUsolve(b)
        # Construct the shape function
        N_i = sum(c * m for c, m in zip(coeffs, monomial_syms))
        shape_functions.append(sp.expand(N_i))

    return shape_functions, nodes

def substitute_barycentric_to_xyz(shape_functions):
    x, y, z = sp.symbols('x y z')
    lambda1 = 1 - x - y - z
    lambda2 = x
    lambda3 = y
    lambda4 = z

    shape_functions_xyz = []
    for N_i in shape_functions:
        N_i_xyz = N_i.subs({L1: lambda1, L2: lambda2, L3: lambda3, L4: lambda4})
        shape_functions_xyz.append(sp.expand(N_i_xyz))
    return shape_functions_xyz

def barycentric_to_cartesian_tetrahedral(nodes_barycentric):
    # Vertices of the tetrahedron
    V1 = (0, 0, 0)
    V2 = (1, 0, 0)
    V3 = (0, 1, 0)
    V4 = (0, 0, 1)
    vertices = [V1, V2, V3, V4]

    nodes_cartesian = []
    for L1, L2, L3, L4 in nodes_barycentric:
        x_node = L1 * V1[0] + L2 * V2[0] + L3 * V3[0] + L4 * V4[0]
        y_node = L1 * V1[1] + L2 * V2[1] + L3 * V3[1] + L4 * V4[1]
        z_node = L1 * V1[2] + L2 * V2[2] + L3 * V3[2] + L4 * V4[2]
        nodes_cartesian.append((x_node, y_node, z_node))
    return nodes_cartesian

def evaluate_shape_functions_tetrahedral(nodes_cartesian, shape_functions):
    num_funcs = len(shape_functions)
    num_nodes = len(nodes_cartesian)
    results = np.zeros((num_funcs, num_nodes))
    x, y, z = sp.symbols('x y z')
    for i, N_i in enumerate(shape_functions):
        for j, (xj, yj, zj) in enumerate(nodes_cartesian):
            N_i_num = N_i.subs({x: xj, y: yj, z: zj})
            results[i, j] = float(N_i_num.evalf())
    return results

def plot_nodes_tetrahedral(nodes_cartesian, degree):
    from mpl_toolkits.mplot3d import Axes3D

    # Unpack x, y, z coordinates
    x_coords = [float(coord[0]) for coord in nodes_cartesian]
    y_coords = [float(coord[1]) for coord in nodes_cartesian]
    z_coords = [float(coord[2]) for coord in nodes_cartesian]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the tetrahedron edges
    vertices = np.array([[0, 0, 0],  # V1
                         [1, 0, 0],  # V2
                         [0, 1, 0],  # V3
                         [0, 0, 1]]) # V4
    edges = [
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3],
        [2, 3]
    ]
    for edge in edges:
        ax.plot(*zip(vertices[edge[0]], vertices[edge[1]]), color='k')

    # Plot the nodes
    ax.scatter(x_coords, y_coords, z_coords, color='red', s=50)

    # Annotate nodes with their indices
    for i, (x_coord, y_coord, z_coord) in enumerate(zip(x_coords, y_coords, z_coords)):
        ax.text(x_coord, y_coord, z_coord, str(i+1), color='blue', fontsize=10)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Nodes of a Degree {degree} Tetrahedral Element')
    plt.show()

# Define degree
degree = 2  # Adjust degree as needed

# Construct shape functions
shape_functions_barycentric, nodes_barycentric = construct_shape_functions_linear_system_tetrahedral(degree)

# Substitute barycentric coordinates with x, y, z
shape_functions = substitute_barycentric_to_xyz(shape_functions_barycentric)

# Convert nodes to Cartesian coordinates
nodes_cartesian = barycentric_to_cartesian_tetrahedral(nodes_barycentric)

# Evaluate shape functions at nodes
evaluation = evaluate_shape_functions_tetrahedral(nodes_cartesian, shape_functions)

# Print evaluation matrix
np.set_printoptions(precision=6, suppress=True)
print("Shape function evaluations at nodes:")
print(evaluation)

# Check if the evaluation matrix is close to identity matrix
identity_check = np.allclose(evaluation, np.identity(len(shape_functions)), atol=1e-12)
print("Is the evaluation matrix close to the identity matrix?", identity_check)

# Plot the nodes
plot_nodes_tetrahedral(nodes_cartesian, degree)

print(nodes_cartesian)

# Display the shape functions
for index, sf in enumerate(shape_functions):
    display(sp.Eq(sp.Symbol(f'N_{index+1}'), sf))
