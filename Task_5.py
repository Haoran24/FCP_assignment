import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import argparse
import sys

class Node:
    def __init__(self, value, index):
        # Initialize each node with a random opinion value and an index
        self.value = value
        self.index = index
        self.connections = []  # List to store connections to other nodes

class Network:
    def __init__(self, size, rewire_prob=0.1, neighbor_range=2):
        # Create a network of nodes in a small-world configuration
        self.nodes = [Node(np.random.uniform(0.45, 0.55), i) for i in range(size)]
        self.size = size
        self.rewire_prob = rewire_prob
        self.neighbor_range = neighbor_range
        self.make_small_world_network()

    def make_small_world_network(self):
        # Establish initial regular ring lattice and then rewire edges with a certain probability
        for node in self.nodes:
            node.connections = [(node.index + i) % self.size for i in range(-self.neighbor_range, self.neighbor_range + 1) if i != 0]
            for i in range(len(node.connections)):
                if random.random() < self.rewire_prob:
                    new_connection = random.randint(0, self.size - 1)
                    while new_connection == node.index or new_connection in node.connections:
                        new_connection = random.randint(0, self.size - 1)
                    node.connections[i] = new_connection

    def update_opinions(self, beta, threshold):
        # Update node opinions based on their neighbors' opinions if difference is under the threshold
        for _ in range(int(len(self.nodes) * 0.5)):  # Update about half of the nodes each iteration
            node = random.choice(self.nodes)
            neighbor_index = random.choice(node.connections)
            neighbor = self.nodes[neighbor_index]
            if abs(node.value - neighbor.value) < threshold:
                delta = beta * (neighbor.value - node.value)
                node.value += delta
                neighbor.value -= delta

    def get_opinions(self):
        # Retrieve current opinions from all nodes
        return [node.value for node in self.nodes]

def plot_network(network, ax_network, mean_opinions, ax_opinion, t):
    # Visualize the network and the evolution of opinions over time
    ax_network.clear()
    ax_opinion.clear()

    G = nx.Graph()
    positions = {node.index: (np.cos(2 * np.pi * node.index / network.size), np.sin(2 * np.pi * node.index / network.size)) for node in network.nodes}
    for node in network.nodes:
        G.add_node(node.index, value=node.value)
        for conn in node.connections:
            G.add_edge(node.index, conn)

    colors = [node['value'] for node in G.nodes.values()]
    nx.draw_networkx_nodes(G, positions, node_color=colors, cmap='viridis', ax=ax_network, node_size=100)
    nx.draw_networkx_edges(G, positions, ax=ax_network, edge_color='k')

    ax_network.set_title("Network State at Iteration {}".format(t))
    ax_network.axis('off')

    ax_opinion.plot(mean_opinions, 'b-')
    ax_opinion.set_title('Mean Opinion Over Time')
    ax_opinion.set_xlabel('Iteration')
    ax_opinion.set_ylabel('Mean Opinion')
    ax_opinion.grid(True)

    plt.pause(0.1)

def run_simulation(size, beta, threshold, max_time, rewire_prob):
    # Setup and execute the simulation
    network = Network(size, rewire_prob=rewire_prob)
    fig, (ax_network, ax_opinion) = plt.subplots(1, 2, figsize=(14, 7))
    mean_opinions = []

    for t in range(max_time):
        network.update_opinions(beta, threshold)
        mean_opinions.append(np.mean(network.get_opinions()))
        if t % 5 == 0:
            plot_network(network, ax_network, mean_opinions, ax_opinion, t)
    plt.show()

def main():
    # Command line argument parsing and launching the simulation
    parser = argparse.ArgumentParser(description='Run the Deffuant model on a small-world network.')
    parser.add_argument('-defuant', action='store_true', help='Flag to run the Deffuant model')
    parser.add_argument('-use_network', type=int, help='Number of nodes in the network')
    parser.add_argument('-beta', type=float, default=0.3, help='Convergence parameter')
    parser.add_argument('-threshold', type=float, default=0.1, help='Threshold for influence')
    parser.add_argument('-max_time', type=int, default=100, help='Number of steps to run the simulation')
    parser.add_argument('-rewire_prob', type=float, default=0.1, help='Probability of rewiring in the small-world network')

    args = parser.parse_args()

    if args.defuant and args.use_network:
        run_simulation(args.use_network, args.beta, args.threshold, args.max_time, args.rewire_prob)
    else:
        print("Error: Missing required arguments. Use -defuant and -use_network with a specified number of nodes.")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

