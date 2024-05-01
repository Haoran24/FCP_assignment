
'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def initialize_opinions(N, seed=None):
    # Initialize the opinions array with random values between 0 and 1
    if seed is not None:
        np.random.seed(seed)  # Set the seed for reproducibility
    return np.random.rand(N)

def update_opinions(opinions, beta, threshold):
    # Update opinions based on the interaction with a randomly chosen neighbor
    N = len(opinions)
    new_opinions = opinions.copy()
    for i in range(N):
        j = (i + np.random.choice([-1, 1])) % N  # Select a random neighbor (left or right)
        if abs(opinions[i] - opinions[j]) < threshold:  # Only interact if within the threshold
            new_opinions[i] += beta * (opinions[j] - opinions[i])  # Adjust opinion
            new_opinions[j] += beta * (opinions[i] - opinions[j])  # Symmetric interaction
    new_opinions = np.clip(new_opinions, 0, 1)  # Ensure opinions stay within [0, 1]
    return new_opinions

def defuant_main(N=100, T=0.2, beta=0.2, max_time=100, seed=None):
    opinions = initialize_opinions(N, seed)
    history = np.zeros((max_time, N))
    for t in range(max_time):
        opinions = update_opinions(opinions, beta, T)
        history[t] = opinions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(opinions, bins=np.linspace(0, 1, N // 2), color='blue')
    plt.xlabel('Opinion')
    plt.ylabel('Frequency')
    plt.title(f'Final Opinion Distribution (T={T}, beta={beta})')
    plt.subplot(1, 2, 2)
    for i in range(N):
        plt.scatter(range(max_time), history[:, i], color='red', s=10)
    plt.xlabel('Time')
    plt.ylabel('Opinion')
    plt.title(f'Opinion Dynamics Over Time (T={T}, beta={beta})')
    plt.tight_layout()
    plt.show()

def test_defuant():
    # Test the model with different beta and threshold values
    for beta in [0.1, 0.3, 0.5]:
        defuant_main(N=100, T=0.2, beta=beta, max_time=100, seed=42)
    for T in [0.1, 0.3, 0.5]:
        defuant_main(N=100, T=T, beta=0.2, max_time=100, seed=42)

def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Run the Deffuant model simulation.')
    # Define flags for command line options
    parser.add_argument('-defuant', action='store_true', help='Run the Deffuant model with default or specified parameters')
    parser.add_argument('-beta', type=float, help='Set the beta value for the model')
    parser.add_argument('-threshold', type=float, help='Set the threshold value for the model')
    parser.add_argument('-test_defuant', action='store_true', help='Run the test functions for the Deffuant model')
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if args.defuant:
        beta = args.beta if args.beta is not None else 0.2
        threshold = args.threshold if args.threshold is not None else 0.2
        defuant_main(N=100, T=threshold, beta=beta, max_time=100, seed=None)
    elif args.test_defuant:
        test_defuant()

if __name__ == "__main__":
    main()
