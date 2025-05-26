# requirements: cvxpy, numpy
import numpy as np
import cvxpy as cp


def test_solve_optimization():
    
    # "unit test 1"
    Z_plus = [[1,1,1]]
    Z_minus = [[0,0,0]]
    problem, v_cl = solve_optimization(Z_plus, Z_minus)
    assert v_cl is not None
    assert problem.status == "optimal"
    assert v_cl.dot(Z_plus[0]) >= v_cl.dot(Z_minus[0])

    # "unit test 2"
    Z_plus = [[1,0,1]]
    Z_minus = [[-1,1,1]]
    problem, v_cl = solve_optimization(Z_plus, Z_minus)
    assert v_cl is not None
    assert problem.status == "optimal"
    assert v_cl.dot(Z_plus[0]) >= v_cl.dot(Z_minus[0])

    # "unit test 3"
    Z_plus = [[1,0,1], [1,1,1]]
    Z_minus = [[0,0,1], [1,0,1]]
    problem, v_cl = solve_optimization(Z_plus, Z_minus)
    assert v_cl is not None
    assert problem.status == "optimal"
    for zp in Z_plus:
        for zm in Z_minus:
            if not v_cl.dot(zp) >= v_cl.dot(zm):
                print("Constraint broken when:")
                print(f"{v_cl.dot(zp)} >= {v_cl.dot(zm)}")
                assert False

    
    # "unit test 4"
    print("unit test 4")
    n = 3
    Z_plus = np.random.rand(100, n)  # |Z_l^+| = 50
    Z_minus = np.random.rand(100, n)  # |Z_l^-| = 50
    problem, v_cl = solve_optimization(Z_plus, Z_minus)
    assert v_cl is not None
    assert problem.status == "optimal"
    for zp in Z_plus:
        for zm in Z_minus:
            if (v_cl.dot(zp) >= v_cl.dot(zm)) or (abs(v_cl.dot(zp) - v_cl.dot(zm)) < 0.00000001):
                pass
            else:
                print("Constraint broken when:")
                print(f"Absolute difference: {abs(v_cl.dot(zp) - v_cl.dot(zm))}")
                print(f"Absolute difference: {abs(v_cl.dot(zp) - v_cl.dot(zm)) < 0.00000001}")
                print(f"{v_cl.dot(zp)} >= {v_cl.dot(zm)}")
                print(f"zp: {zp}")
                print(f"zp: {zm}")
                assert False


    # "unit test 5"
    print("unit test 5")
    n = 3
    Z_plus = np.random.rand(50, n)  # |Z_l^+| = 50
    Z_minus = np.random.rand(50, n)  # |Z_l^-| = 50
    problem, v_cl = solve_optimization(Z_plus, Z_minus)
    assert v_cl is not None
    assert problem.status == "optimal"
    for zp in Z_plus:
        for zm in Z_minus:
            if not v_cl.dot(zp) >= v_cl.dot(zm):
                print("Constraint broken when:")
                print(f"{v_cl.dot(zp)} >= {v_cl.dot(zm)}")
                print(f"zp: {zp}")
                print(f"zp: {zm}")
                assert False






def solve_optimization(Z_plus: list, Z_minus: list):
    n = 3 # dimensionality of vectors

    constraint_vector = []
    v_cl = cp.Variable(n) # make the v_cl as long as the input vectors
    for zp in Z_plus:
        for zm in Z_minus:
            constraint = v_cl @ zp >=  v_cl @ zm
            constraint_vector.append(constraint)
    if len(Z_plus) < 10:
        print(f"Solved, for Z_plus: {Z_plus}")
        print(f"Solved, for Z_minus: {Z_minus}")


    objective = cp.Minimize(cp.norm1(v_cl))


    problem = cp.Problem(objective, constraint_vector)
    problem.solve()

    return problem, v_cl.value
    print(problem.status)
    print(problem.value)
    print(v_cl.value)
    # print(Z_plus)

if __name__ == "__main__":
    test_solve_optimization()


# Conceptual code for extracting concept vectors from Lc0
def extract_concept_vectors(network, positions, mcts_searches):
    # Find optimal and suboptimal rollouts from MCTS
    optimal_rollouts, suboptimal_rollouts = analyze_mcts_statistics(mcts_searches)
    
    # Extract latent representations for these rollouts
    optimal_representations = get_representations(network, optimal_rollouts)
    suboptimal_representations = get_representations(network, suboptimal_rollouts)
    
    # Use convex optimization to find concept vectors
    concept_vectors = []
    for i in range(num_concepts_to_extract):
        v = solve_optimization(optimal_representations, suboptimal_representations)
        concept_vectors.append(v)
        
    return concept_vectors
