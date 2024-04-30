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
        return np.where(np.array(self.connections)==1)[0]

class Queue:
    def __init__(self):
        self.queue = []

    def push(self, item):
        self.queue.append(item)

    def pop(self):
        if self.is_empty():
            return None
        return self.queue.pop(0)

    def is_empty(self):
        return len(self.queue) == 0

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
        #Loop through each nodes in the network
        total_degree = sum(sum(node.connections) for node in self.nodes)
        mean_degree = total_degree/ len(self.nodes)
        return mean_degree

    def get_mean_clustering(self):
        '''
        Calculate the mean clustering co-efficient
        (the fraction of a node's neighbours forming a triangle that includes the original node)
        '''
        #List of the clustering coefficient of each nodes
        clustering_coefficient = []

        #Loop through each nodes
        for node in self.nodes:
            #Find neighbours to each nodes
            neighbours = node.get_neighbours()

            #Number of neighbour
            num_neighbours = len(neighbours)
            
            #Formula for the number of possible triangles that can be formed
            possible_triangles = num_neighbours * (num_neighbours-1) / 2

            connected_nodes = sum(node.connections[neighbour] for neighbour in neighbours)
            triangle = int(possible_triangles/ connected_nodes)

            clustering_coefficient.append(triangle)
                
        mean_clustering_coefficient = np.mean(clustering_coefficient)
        return mean_clustering_coefficient

    # def bfs(self, start_node, goal):
    #     start_node = self.graph[0]
    #     print("start:", start_node)
    #     goal = self.nodes[-1]
    #     print("goal: ", goal)
    #     search_queue = Queue()
    #     search_queue.push(start_node)
    #     visited = []
        
    #     while not search_queue.is_empty():
    #         #Pop the next node from the Queue
    #         node_to_check = search_queue.pop()
    #         #If we are at the goal, then we are finished.
    #         if node_to_check == goal:
    #             break
            
    #         for neighbour_index in node_to_check.get_neighbours():
    #             #Get a node based on the index
    #             neighbour = self.nodes[neighbour_index]
    #             if neighbour_index not in visited:
    #                 search_queue.push(neighbour)
    #                 visited.append(neighbour_index)
    #                 neighbour.parent = node_to_check
                 
    #     node_to_check = goal
    #     start_node.parent = None
    #     route = []
    #     while node_to_check.parent:
    #         route.append(node_to_check)
    #         node_to_check = node_to_check.parent
        
    #     route.append(node_to_check)
    def bfs(self, start_node, goal):
        search_queue = Queue()
        search_queue.push(start_node)
        visited = set()
        visited.add(start_node)  # Mark the start node as visited
        distance = {start_node: 0}  # Initialize the distance dictionary with the start node
                 
        while not search_queue.is_empty():
            node_to_check = search_queue.pop()
            if node_to_check == goal:
                return distance[node_to_check]
            
            for neighbour_index in node_to_check.get_neighbours():
                neighbour = self.nodes[neighbour_index]
                if neighbour not in visited:
                    visited.add(neighbour)
                    distance[neighbour] = distance[node_to_check] + 1  # Update the distance for the neighbour
                    search_queue.push(neighbour)
                    
        return distance
                
    def get_mean_path_length(self):
        '''
        Calculate the mean path length
        (average of the distance between two nodes)
        '''
        total_path_length = 0
        num_pairs = 0
        
        for node in self.nodes:
            neighbours = node.get_neighbours()
            print("Node:", node.index)
            print("neighbours: ", neighbours)
            for neighbour_index in neighbours:
                print("index:", neighbour_index)
                path_length = self.bfs(node, self.nodes[neighbour_index])
                print("PL:", path_length)
                if path_length is not None:
                    total_path_length += path_length
                    print("TPL:", total_path_length)
                    num_pairs += 1
                    print("Num Pairs: ", num_pairs)
                    
        if num_pairs == 0:
            return 0  # No pairs of nodes found, return 0 as mean path length
        else:
            return total_path_length / num_pairs
        
    def make_random_network(self, N, connection_probability):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''
        self.nodes = []
        
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))
            
        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
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
            
            circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)
            
            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)
                    
                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

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
    assert(network.get_mean_degree()==2), network.get_mean_degree()
    assert(network.get_mean_clustering()==0), network.get_mean_clustering()
    assert(network.get_mean_path_length()==2.777777777777778), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert(network.get_mean_degree()==1), network.get_mean_degree()
    assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
    assert(network.get_mean_path_length()==5), network.get_mean_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
    assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
    assert(network.get_mean_path_length()==1), network.get_mean_path_length()

    print("All tests passed")

def main():
    parser = argparse.ArgumentParser(description="Generate and analyse networks")
    parser.add_argument("-network",  type=int, help="Generate and analyze a random network of the specified size")
    parser.add_argument("-test_network",action="store_true", help="Run test functions for network metrics")
    
    args = parser.parse_args()
    print(args.network)
    print(args.test_network)
    
    if args.network:
        network = Network()
        network.make_random_network(10, 0.5)
        
        mean_degree = network.get_mean_degree()
        mean_clustering = network.get_mean_clustering()
        mean_path_length = network.get_mean_path_length()
        
        print("Mean degree: ", mean_degree)
        print("Average path length: ", mean_path_length)
        print("Clustering co-efficient: ", mean_clustering)
        
        network.plot()
        plt.show()
    
    elif args.test_network:
        test_networks()
        
if __name__ == "__main__":
    test_networks()
