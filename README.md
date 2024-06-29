# FCP_assginment
FCP_final_assignment

Github Repo:
 https://github.com/Haoran24/FCP_assignment

-- Task_1-4 in the assignment.py
  (Check specific task at Github branches)
-- Task_5 in the Task_5.py


Task1：
Overview：
This Python program simulates the Ising model of opinion dynamics. This model considers a population where individuals can hold opinions either for or against an issue, represented by values of +1 or -1. Individuals prefer to align with their neighbors' opinions, and the model calculates the level of agreement or disagreement based on immediate neighbor interactions. The model can be influenced by external factors and individual tolerance levels for disagreement.

Installation：
NumPy library
Matplotlib library
Ensure you have the required libraries installed using pip: pip install numpy matplotlib
Argparse

Usage：
To run the simulation with default parameters, execute:
$ python assignment.py -ising-model

To include external influence or adjust the tolerance level (alpha):
$ python assignment.py -ising_model -external -0.1
$ python assignment.py -ising_model -alpha 10

To run the test functions to verify the correctness of the implementation:
$ python assignment.py -test_ising

Model Parameters:
Alpha (α): Controls the tolerance level for disagreement. A higher alpha means lower tolerance, reducing the likelihood of opinion change.
External (H): Represents external influence strength. Positive values favor one opinion, while negative values favor the opposite.

Output:
The program outputs the following plots:

Histogram: Shows the distribution of opinions at the end of the simulation.
Animation: Displays how opinions evolve over time across the population grid.



Task2：
Overview：
This Python program simulates the Deffuant model of opinion dynamics. The model represents opinions on a continuous scale from 0 to 1, where individuals update their opinions through interactions with their neighbors, subject to a certain threshold for interaction.


Installation：
NumPy library
Matplotlib library
Ensure you have the required libraries installed using pip: pip install numpy matplotlib
Argparse

Usage：
To run the simulation with default parameters, execute the script without any flags:
$ python assignment.py -defuant

To specify the beta parameter (coupling parameter) or the threshold for interaction, use the flags '-beta' and '-threshold':
$ python assignment.py -defuant -beta 0.1
$ python assignment.py -defuant -threshold 0.3

To run the test functions to verify the correctness of the implementation:
$ python assignment.py -test_defuant


Model Parameters:
beta: Coupling parameter, which determines the influence of neighbors' opinions. If beta is large, each update will cause stronger shifts towards the mean opinion.
threshold: The threshold for interaction, which limits the agreement to people whose opinion is within some distance of the user's own opinion.


Output:
The program outputs two plots:
1.A histogram showing the final distribution of opinions among the population.
2.A scatter plot visualizing the dynamics of opinions over time.



Task3：
Installation:
NumPy library
Matplotlib library
Ensure you have the required libraries installed using pip: pip install numpy matplotlib
Argparse

Overview:
This Python program creates a random plot of network and caluclate three parameters which are the mean degree, mean path length and mean clustering co-efficient

Usage:
To run the simulation with default parameters, execute the script:
$ python3 assignment.py -network 10

To run the test functions to verify the correctness of the implementation:
$ python3 assignment.py -test_network

Output:
The program outputs the following plots:

Dot line diagram: shows the network image of the random network formed

Task4：
Overview：
This Python program simulates the ring network  model of opinion dynamics.Small world networks are charactised by having higher clustering co-efficients and lower mean path lengths than random networks of a similar size set rewire probability and produce a ring network

Installation：
NumPy library
Matplotlib library
Ensure you have the required libraries installed using pip: pip install numpy matplotlib
Argparse

Usage：
python3 assignment.py -ring_network 10 # This should create a ring network with a range of 1 and a size of 10
 $ python3 assignment.py -small_world 10 # This should create a small-worlds network with default parameters
 $ python3 assignment.py -small_world 10 -re_wire 0.1 #This should create a small worlds network with a re-wiring probability of 0

Output:
The program outputs the following plots:

Dot line diagram: shows the network image from one to N and the possibility of connecting two dots in a small world.



Task5：
Open Task_5.py to check this code

Overview：
This project includes a Python script that simulates the evolution of mean opinions over time within a network of individuals or nodes. The opinions are influenced by the average opinion of the network and random fluctuations, which might simulate personal changes or external influences. The main objective is to observe how opinions stabilize or vary over a given number of iterations.

Installation：
matplotlib for plotting graphs.
numpy for numerical operations.
Networkx
Ensure you have the required libraries installed using pip: pip install numpy matplotlib Network

Usage：
The script opinion_simulation.py can be easily configured by adjusting the following parameters in the script:
iterations: Total number of iterations to simulate. Default is 30.
size: Number of nodes (or agents) in the network. Default is 100.

Run：
$ python Task_5.py -defuant -use_network 10
