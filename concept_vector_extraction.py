# requirements: cvxpy, numpy
import numpy as np
import cvxpy as cp


def solve_optimization():
    n = 3 # dimensionality of vectors
    Z_plus = np.random.randn(50, n)  # |Z_l^+| = 50
    Z_minus = np.random.randn(50, n)  # |Z_l^-| = 50
    # ^for testing the solver code
    print(Z_plus)

if __name__ == "__main__":
    solve_optimization()


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
