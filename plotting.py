import pandas as pd
import matplotlib.pyplot as plt

def create_plot(folder, save_png=False):
    """
    Creates a plot of the fitness vs generation for the given folder.
    One line for the best fitness, one for the mean fitness and a band for the standard deviation.
    """
    # Get enemy numbers
    enemies = folder.split('_')[2]
    print(enemies)
    
    # Read data from csv file
    df = pd.read_csv(folder + '/results.csv')
    
    # Plot gen vs best fitness
    plt.figure(figsize=(10, 5))
    plt.plot(df['gen'], df['best'], color='green')
    plt.plot(df['gen'], df['mean'], color='orange')
    plt.fill_between(df['gen'], df['mean'] - df['std'], df['mean'] + df['std'], alpha=0.5, color='orange')

    # Add legend
    plt.legend(['Best', 'Mean', 'Std'])

    plt.title('Fitness vs generation for enemies: ' + enemies)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    if save_png:
        plt.savefig(folder + '/plot.png')
    plt.show()
