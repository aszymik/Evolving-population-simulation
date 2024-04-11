import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from random import choice

# TODO:
# rozmiary / osie dopasować
# zmienić funkcję reprodukcji – mutacje przy powstawaniu nowych osobników?
# boxploty cech
# obsłużyć wymieranie populacji
# macierz PCA do wyliczania promieni koła – bierzemy dwa pierwsze wiersze z macierzy loadings
# punkty na okręgu muszą spełniać rownanie exp(u-opt)... = fitness_threshold – z tego próbkujemy


def rotate(vector, angle):
  angle = 2 * np.pi * angle / 360
  x = vector[0] * np.cos(angle) - vector[1] * np.sin(angle)
  y = vector[0] * np.sin(angle) + vector[1] * np.cos(angle)
  return np.array([x, y])


class Population:

  def __init__(self, N, max_N, n, env_change, T, mutation_prob, mutation_std,
               fitness_std, reproduction_thr, max_num_children, angle):
    self.N = N
    self.max_N = max_N
    self.n = n
    self.env_change = env_change  # stała w update'cie optymalnego genomu
    self.T = T  # czas kiedy zachodzi duża zmiana w środowisku
    self.mutation_prob = mutation_prob
    self.mutation_std = mutation_std
    self.optimal_genotypes = [np.random.normal(0, 0.05, size=self.n)]
    self.population = self.initialize_population()
    self.reproduction_threshold = reproduction_thr,
    self.max_num_children = max_num_children
    self.angle = angle
    self.fitness_threshold = self.fitness_threshold = len(
        self.population) / (len(self.population) + self.max_N//2)
    
    self.offspring_thresholds = np.linspace(self.fitness_threshold, 1, num=self.max_num_children+1)
    self.fitness_std = fitness_std
    self.pca = PCA(n_components=2)
    self.pca.fit(self.population)
    

  def initialize_population(self):
    return np.array([
        self.optimal_genotypes[0] + np.random.normal(0, 0.5, self.n)
        for _ in range(self.N)
    ])

  def fitness_function(self, organism: np.ndarray):
    # Oblicza fitness w stosunku do każdego z optymalnych genotypów i zwraca największy
    fitness = []
    for opt in self.optimal_genotypes:
      fitness.append(
          np.exp(-np.linalg.norm(organism - opt)**2 /
                 (2 * self.fitness_std**2)))
    return max(fitness)

  def mutation(self, organism: np.ndarray):
    if np.random.uniform() < self.mutation_prob:
      mutation_index = np.random.randint(0, self.n)
      organism[mutation_index] += np.random.normal(0, self.mutation_std)
    return organism

  def selection(self) -> None:
    # Eliminowanie osobników z dopasowaniem poniżej średniej
    fitness_values = np.array(
        [self.fitness_function(organism) for organism in self.population])

    self.population = np.array([
        organism for organism, fitness in zip(self.population, fitness_values)
        if fitness >= self.fitness_threshold
    ])

    # Redukcja populacji do N osobników
    if len(self.population) > self.max_N:
      np.random.shuffle(self.population)
      self.population = self.population[:self.N]

  def reproduction(self) -> None:
    # Obliczanie dostosowania dla każdego osobnika w populacji
    fitness_scores = np.array(
        [self.fitness_function(organism) for organism in self.population])

    # Podział na 8 grup na podstawie dostosowania
    offspring_thresholds = np.linspace(self.fitness_threshold,
                                       1,
                                       num=self.max_num_children + 1)
    self.offspring_thresholds = offspring_thresholds
    offspring_groups = np.digitize(fitness_scores,
                                   offspring_thresholds,
                                   right=True)
    # Przypisanie liczby dzieci dla każdego osobnika w zależności od fitness
    num_offspring_by_group = np.arange(0, self.max_num_children + 1)

    # Reprodukcja z uwzględnieniem liczby dzieci
    offspring = []
    for group, num_offspring in zip(offspring_groups, num_offspring_by_group):
      group_indices = np.where(offspring_groups == group)[0]
      for parent_index in group_indices:
        for _ in range(num_offspring):
          offspring.append(self.population[parent_index].copy())

    # Aktualizacja populacji
    self.population = np.array(offspring)  # osobnik żyje tylko jedno pokolenie

  def environment_change(self, generation):
    if generation != 0 and generation % self.T == 0:  # kolejne optimum
      optim_idx = np.random.randint(len(self.optimal_genotypes))
      new = rotate(self.optimal_genotypes[optim_idx], angle=-self.angle)
      self.optimal_genotypes[optim_idx] = rotate(
          self.optimal_genotypes[optim_idx], angle=self.angle)
      self.optimal_genotypes.append(new)
    else:
      self.optimal_genotypes = [
          opt + self.env_change * opt for opt in self.optimal_genotypes
      ]    

  def evolve(self, generation):
    self.population = np.array(
        [self.mutation(organism) for organism in self.population])
    self.selection()
    self.reproduction()
    self.environment_change(generation)
    self.fitness_threshold = len(self.population) / (len(self.population) + self.max_N//2)

  def simulation(self, generations):
    fig, ax = plt.subplots()

    population = np.array(self.population)
    optimal = np.array(self.optimal_genotypes)

    population_plot = ax.scatter(population[:, 0],
                                 population[:, 1],
                                 label='Population',
                                 marker='.',
                                 s = 1,
                                 alpha = 0.8,
                                 zorder=2)
    optimal_plot = ax.scatter(optimal[:, 0],
                              optimal[:, 1],
                              label='Optimal genotypes',
                              marker='.',
                              s = 7,
                              color='orange',
                              alpha=0.8,
                              zorder=3)
    radii = []
    for thr in self.offspring_thresholds:
      dist = -2 * self.fitness_std**2 * np.log(thr) if thr != 0 else 0
      radii.append(dist)

    circles_plot = []
    for i in range(len(optimal)):
      for j in range(len(radii)):
        circle = plt.Circle((optimal[i,0], optimal[i,1]), radii[j], 
                            color='green', 
                            alpha=(j+1)/len(radii), 
                            label=f'{j} children')
        if j == 1:
          circle.set_label('1 child')
        circles_plot.append(circle)

    for circle in circles_plot:
        ax.add_artist(circle)  
    
    
    ax.legend(fontsize=8)
    fig.suptitle(f'Generation 0')
    ax.set_title(f'Number of organisms: {len(self.population)}', fontsize=11)

    def update_plot(frame):
      self.evolve(frame)
      population = np.array(self.population)
      optimal = np.array(self.optimal_genotypes)

      population_plot.set_offsets(population)
      optimal_plot.set_offsets(optimal)

      for circle in circles_plot:
        circle.remove()
    
      circles_plot.clear()
      for i in range(len(optimal)):
          for j in range(len(radii)):
            circle = plt.Circle((optimal[i,0], optimal[i,1]), radii[j], alpha=(j+1)/len(radii), color='green')
            ax.add_artist(circle)
            circles_plot.append(circle)

      fig.suptitle(f'Generation {frame}')
      ax.set_title(f'Number of organisms: {len(self.population)}', fontsize=11)

      return (population_plot, optimal_plot)

    ani = FuncAnimation(fig=fig,
                        func=update_plot,
                        frames=generations+1,
                        interval=200,
                        repeat=False)
    plt.show()


# Przykład użycia
population = Population(N=200,
                        max_N=1000,
                        n=2,
                        env_change=0.02,
                        T=50,
                        mutation_prob=0.75,
                        mutation_std=0.3,
                        fitness_std=0.2,
                        reproduction_thr=0.5,
                        max_num_children=7,
                        angle=30)
population.simulation(generations=155)
