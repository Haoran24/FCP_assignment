import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse


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
    #
    change_in_agreement = external * population[row, col]
    # Checking neighbours in all four directions
    for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        n_row, n_col = row + d_row, col + d_col
        if 0 <= n_row < n_rows and 0 <= n_col < n_cols:
            change_in_agreement += population[n_row, n_col] * population[row, col]
    return change_in_agreement

def ising_step(population, alpha=1.0, external=0.0):
    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)

    change_in_agreement = calculate_agreement(population, row, col, external)

    # Calculate the flip probability inversely proportional to alpha
    p_flip = np.exp(-abs(change_in_agreement) * alpha)

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

    for frame in range(100):
        for step in range(1000):
            ising_step(population, alpha, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)
    plt.show()

# This function should be defined within your '1.py' script

def main():
    parser = argparse.ArgumentParser(description='Run the Ising model simulations with various settings.')
    parser.add_argument('-ising_model', action='store_true', help='Run the Ising model with default settings')
    parser.add_argument('-external', type=float, default=0.0, help='Specify the strength of external influence')
    parser.add_argument('-alpha', type=float, default=1.0, help='Specify the alpha value for agreement calculation')
    parser.add_argument('-test_ising', action='store_true', help='Run the Ising model test functions')

    args = parser.parse_args()

    if args.alpha <= 0:
        raise ValueError("The alpha parameter must be greater than 0.")

    if args.test_ising:
        test_ising()
    elif args.ising_model:
        T = 1.0 / args.alpha if args.alpha != 0 else float('inf')
        H = args.external
        population = np.random.choice([-1, 1], size=(100, 100))
        ising_main(population, T, H)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()


















