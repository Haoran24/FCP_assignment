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
        #Loop through each nodes in the network and calculate the sum of the degree of each nodes
        total_degree = sum(sum(node.connections) for node in self.nodes)
        #Calculate the mean degree of the nodes in the network
        mean_degree = total_degree/ len(self.nodes)
        return mean_degree

    def get_mean_clustering(self):
        '''
        Calculate the mean clustering co-efficient
        (the fraction of a node's neighbours forming a triangle that includes the original node)
        '''
        #Empty list of the clustering coefficient of each nodes
        clustering_coefficient = []

        #Loop through each nodes
        for node in self.nodes:
            #Find neighbours to each nodes
            neighbours = node.get_neighbours()

            #Number of neighbour
            num_neighbours = len(neighbours)
            
            #Formula for the number of possible triangles that can be formed
            possible_triangles = num_neighbours * (num_neighbours-1) / 2

            #Loop through each neighbour of the nodes to find the number of nodes connected to the node
            connected_nodes = sum(node.connections[neighbour] for neighbour in neighbours)
            
            #clustering coefficient for each nodes
            triangle = int(possible_triangles/ connected_nodes)

            #Add clustering coefficient of each node in the clustering_coefficient empty list above
            clustering_coefficient.append(triangle)
        
        #Calculate the average clustering coefficient of the whole network
        mean_clustering_coefficient = np.mean(clustering_coefficient)
        return mean_clustering_coefficient
    
    def floyds_algo(self):
        
        num_nodes = len(self.nodes)
        
        #INF represents infinity, which is used to denote unreachable vertices
        INF = float('inf')
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i!=j:
                    path_length = INF
                
                else:
                    path_length = 0
        
        #Initialise distance to direct edges
        for node in self.nodes:
            for neighbour_index, connected in enumerate(node.connections):
                if connected:
                    path_length[node.index][neighbour_index] = 1
        
        #Update distance matrix
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if path_length[i][k] + path_length[k][j] < path_length[i][j]:
                        path_length[i][j] = path_length[i][k] + path_length[k][j]
        
        return path_length
        
    def get_mean_path_length(self):
        num_nodes = len(self.nodes)
        total_path_length = 0
        num_paths = 0
        
        path_dist = self.floyds_algo(self.nodes)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i!=j and path_dist[i][j]!=INF:
                    total_path_length+=path_dist[i][j]
                    num_paths +=1
                    
        if num_paths==0:
            return INF
        
        average_path_length = total_path_length/num_paths
        
        return(average_path_length, 15)


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
    main()

