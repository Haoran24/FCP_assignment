import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse


class Node:
    '''
    Class to represent a node in a network
    '''

    def __init__(self, value, number, connections=None):
        self.index = number
        self.connections = connections
        self.value = value

    def get_neighbours(self):
        '''
        Neighbours is a numpy array representing the row of the adjacency matrix that corresponds to the node
        '''
        return np.where(np.array(self.connections) == 1)[0]


class Network:

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []

        else:
            self.nodes = nodes

    def get_mean_degree(self):
        '''
        Calculate the mean degree of the network
        '''
        # Loop through each nodes in the network and calculate the sum of the degree of each nodes
        total_degree = sum(sum(node.connections) for node in self.nodes)
        # Calculate the mean degree of the nodes in the network
        mean_degree = total_degree / len(self.nodes)
        return mean_degree

    def get_mean_clustering(self):
        '''
        Calculate the mean clustering co-efficient
        (the fraction of a node's neighbours forming a triangle that includes the original node)
        '''
        # Empty list of the clustering coefficient of each nodes
        clustering_coefficient = []

        # Loop through each nodes
        for node in self.nodes:
            # Find neighbours to each nodes
            neighbours = node.get_neighbours()

            # Number of neighbour
            num_neighbours = len(neighbours)

            # Skip nodes with fewer than 2 neighbours
            if num_neighbours < 2:
                continue

            num_triangles = 0

            for i in range(num_neighbours):
                for j in range(i + 1, num_neighbours):
                    if self.nodes[neighbours[i]].connections[neighbours[j]] == 1:
                        num_triangles += 1

            # Formula for the number of possible triangles that can be formed
            possible_triangles = num_neighbours * (num_neighbours - 1) / 2

            node_cluster = num_triangles / possible_triangles
            clustering_coefficient.append(node_cluster)

        mean_clustering_coefficient = sum(clustering_coefficient) / len(self.nodes)

        return mean_clustering_coefficient

    def floyds_algo(self):
        '''
        Use Floyd-Warshall algorithm to  find the path length
        '''
        # Number of nodes
        num_nodes = len(self.nodes)

        # INF represents infinity, which is used to denote unreachable vertices
        INF = float('inf')

        # Length of each path
        path_length = np.zeros((num_nodes, num_nodes))

        # Set the range for path length
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    path_length[i][j] = INF

                else:
                    path_length[i][j] = 0

        # Initialise distance to direct edges
        for node in self.nodes:
            for neighbour_index, connected in enumerate(node.connections):
                if connected:
                    path_length[node.index][neighbour_index] = 1

        # Update distance matrix
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if path_length[i][k] + path_length[k][j] < path_length[i][j]:
                        path_length[i][j] = path_length[i][k] + path_length[k][j]

        return path_length

    def get_mean_path_length(self):
        '''
        Calculate the mean path length
        (average of the distance between two nodes)
        '''
        num_nodes = len(self.nodes)
        # Initialise path length
        total_path_length = 0
        num_paths = 0

        # INF represents infinity, which is used to denote unreachable vertices
        INF = float('inf')

        path_dist = self.floyds_algo()

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and path_dist[i][j] != INF:
                    total_path_length += path_dist[i][j]
                    num_paths += 1

        if num_paths == 0:
            return INF

        # Calculate the average path length
        average_path_length = total_path_length / num_paths

        # Round the average path length to 15 d.p
        return (round(average_path_length, 15))

    def make_random_network(self, N, connection_probability):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            # creat node with a random value without connection ,then put that into list of nodes
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            # select the next node and use probability to connect together
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=1):
        # connect node each other
        self.nodes = []
        for node_number in range(N):
            # creat node with a random value without connection ,then put that into list of nodes
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            # select the smaller neighbour_index and connect
            for neighbour_index in range(index + 1, N):
                if abs(index - neighbour_index) % (N-neighbour_range) <= neighbour_range:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_small_world_network(self, N, re_wire_prob=0.2):
        # creat small network with n and re_wire_prob
        neighbour_range=2
        self.nodes = []
        for node_number in range(N):
            # creat node with a random value without connection ,then put that into list of nodes
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):

            for neighbour_index in range(index + 1, N):
                if abs(index - neighbour_index) % (N-neighbour_range) <= neighbour_range:
                    if np.random.random() < re_wire_prob:
                        neighbour_index = np.random.choice (range(index+1,N))
                        node.connections[neighbour_index] = 1
                        self.nodes[neighbour_index].connections[index] = 1
                    else:
                        node.connections[neighbour_index] = 1
                        self.nodes[neighbour_index].connections[index] = 1
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 1.2*num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
        plt.show()
def test_networks():

    #Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number-1)%num_nodes] = 1
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)
    print("Testing ring network")
    # assert(network.get_mean_degree()==2), network.get_mean_degree()
    # assert(network.get_clustering()==0), network.get_clustering()
    # assert(network.get_path_length()==2.777777777777778), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    # assert(network.get_mean_degree()==1), network.get_mean_degree()
    # assert(network.get_clustering()==0),  network.get_clustering()
    # assert(network.get_path_length()==5), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    # assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
    # assert(network.get_clustering()==1),  network.get_clustering()
    # assert(network.get_path_length()==1), network.get_path_length()

    print("All tests passed")


'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the extent to which a cell agrees with its neighbours.
    Inputs: population (numpy array)
    		row (int)
    		col (int)
    		external (float)
    Returns:
    		change_in_agreement (float)
    '''

    n_rows, n_cols = population.shape
    # Opinion
    current_cell_state = population[row, col]
    # Start with the influence of external factor
    change_in_agreement = external * current_cell_state

    # Define neighbor offsets for four directions: up, down, left, right
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Calculate agreement with neighbors
    for d_row, d_col in neighbor_offsets:
        neighbor_row, neighbor_col = row + d_row, col + d_col
        if 0 <= neighbor_row < n_rows and 0 <= neighbor_col < n_cols:
            neighbor_state = population[neighbor_row, neighbor_col]
            change_in_agreement += neighbor_state * current_cell_state

    return change_in_agreement


def ising_step(population, alpha=1.0, external=0.0):
    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    change_in_agreement = calculate_agreement(population, row, col, external)

    # Calculate the flip probability proportional to alpha
    p_flip = np.exp(-abs(change_in_agreement) / alpha)

    # Flip based on agreement and random chance influenced by alpha
    if change_in_agreement < 0 or np.random.rand() < p_flip:
        population[row, col] *= -1


def plot_ising(im, population):
    """
    Displays the current state of the Ising model.
    Parameters.
        im: previously created matplotlib image object.
        population: numpy array representing the current state of the population grid.
    """
    # Update the image data
    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    # Pause to update the image
    plt.pause(0.1)

def test_ising():
    """
    Testing the calculate_agreement function of the Ising model
    """
    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    # Testing the impact of external opinions
    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population, 1, 1, 1)==3), "Test 7"
    assert(calculate_agreement(population, 1, 1, -1)==5), "Test 8"
    assert (calculate_agreement(population, 1, 1, 10) ==-6), "Test 9"
    assert (calculate_agreement(population, 1, 1, -10) ==14), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, alpha, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)
    plt.show()


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

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

'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''


def main():
    # Setting up the argument parser
    parser = argparse.ArgumentParser(description='Run various models and network simulations with adjustable settings.')

    # Arguments for Ising and Deffuant models
    parser.add_argument('-ising_model', action='store_true', help='Run the Ising model with default or specified settings')
    parser.add_argument('-external', type=float, default=0.0, help='Specify the strength of external influence on the model')
    parser.add_argument('-alpha', type=float, default=1.0, help='Specify the alpha value, controlling the impact of agreement on flipping probability')
    parser.add_argument('-test_ising', action='store_true', help='Run the test functions for the Ising model')
    parser.add_argument('-defuant', action='store_true', help='Run the Deffuant model with default or specified parameters')
    parser.add_argument('-beta', type=float, default=0.2, help='Set the beta value for the model')
    parser.add_argument('-threshold', type=float, default=0.2, help='Set the threshold value for the model')
    parser.add_argument('-test_defuant', action='store_true', help='Run the test functions for the Deffuant model')

    # Arguments for network simulations
    parser.add_argument('-network', type=int, help="Generate and analyze a random network of the specified size")
    parser.add_argument('-test_network', action='store_true', help="Run test functions for network metrics")
    parser.add_argument('-ring_network', type=int, help='Create a ring network with the specified size')
    parser.add_argument('-small_world', type=int, help='Create a small-worlds network with default parameters')
    parser.add_argument('-re_wire', type=float, default=0.2, help='Re-wiring probability for the small-world network')

    args = parser.parse_args()

    # Initialize the network object if any network related operation is requested
    if any([args.network, args.ring_network, args.small_world, args.test_network]):
        network = Network()

    # Model handling
    if args.test_ising:
        test_ising()
    elif args.test_defuant:
        test_defuant()
    elif args.ising_model:
        if args.alpha <= 0:
            raise ValueError("The alpha parameter must be greater than 0.")
        population = np.random.choice([-1, 1], size=(100, 100))
        ising_main(population, alpha=args.alpha, external=args.external)
    elif args.defuant:
        beta = args.beta if args.beta is not None else 0.2
        threshold = args.threshold if args.threshold is not None else 0.2
        defuant_main(N=100, T=threshold, beta=beta, max_time=100, seed=None)

    # Network handling
    if args.network:
        network.make_random_network(args.network, 0.5)
        print("Mean degree: ", network.get_mean_degree())
        print("Average path length: ", network.get_mean_path_length())
        print("Clustering co-efficient: ", network.get_mean_clustering())
        network.plot()
    elif args.ring_network:
        network.make_ring_network(args.ring_network)
        network.plot()
    elif args.small_world:
        network.make_small_world_network(args.small_world, args.re_wire)
        network.plot()
    elif args.test_network:
        test_networks()

    if not any([args.ising_model, args.defuant, args.network, args.ring_network, args.small_world, args.test_ising, args.test_defuant, args.test_network]):
        parser.print_help()

if __name__ == "__main__":
    main()


