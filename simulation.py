import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

# TODO:
# ogarnąć wymiary okręgów (PCA)
# macierz PCA do wyliczania promieni koła – bierzemy dwa pierwsze wiersze z macierzy loadings
# punkty na okręgu muszą spełniać rownanie exp(u-opt)... = fitness_threshold – z tego próbkujemy
# legenda
# ile osobników w danym momencie
# widoczność zeby organizmy były "u góry"
# ew. jeśli nie ma osobników przy danym optimum to go nie printować ale za duzo roboty chyba


def rotate(vector, angle):
    angle = 2 * np.pi * angle / 360
    x = vector[0] * np.cos(angle) - vector[1] * np.sin(angle)
    y = vector[0] * np.sin(angle) + vector[1] * np.cos(angle)
    return np.array([x, y])


class Population:

    def __init__(self, N, max_N, n, env_change, T, mutation_prob, mutation_std,
                   fitness_std, fitness_thr, max_num_children, angle):
        self.N = N
        self.max_N = max_N
        self.n = n
        self.env_change = env_change  # stała w update'cie optymalnego genomu
        self.T = T  # czas kiedy zachodzi duża zmiana w środowisku
        self.mutation_prob = mutation_prob
        self.mutation_std = mutation_std
        self.optimal_genotypes = [np.random.normal(0, 0.05, size=self.n)]
        self.population = self.initialize_population()
        self.max_num_children = max_num_children
        self.angle = angle
        # self.fitness_threshold = len(
        #     self.population) / (len(self.population) + self.max_N//2)
        self.fitness_threshold = fitness_thr
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
            fitness.append(np.exp(-np.linalg.norm(organism - opt)**2 / (2 * self.fitness_std**2)))
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
        
        offspring_groups = np.digitize(fitness_scores,
                                       self.offspring_thresholds,
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

    def optima_df(self, x: int, y: int, gen: int) -> pd.DataFrame:
        df = pd.DataFrame(columns=['x', 'y', 'generation', 'radius', 'type'])

        for thr in self.offspring_thresholds:
            r = -2 * self.fitness_std ** 2 * np.log(thr) if thr != 0 else 0
            df.loc[len(df)] = [x, y, gen, r, 'optimum']
        df['generation'] = df['generation'].astype(int)
        return df

    def simulation(self, generations: int) -> pd.DataFrame:
        # 1. tworzenie pustego data frama do organizmów z odpowiednimi kolumnami (x, y, generation)
        # 2. tworzenie pustego data frame do optimów z kolumnami (x, y, generation, radius)

        # pca_population = self.pca.transform(self.population)
        # pca_optima = self.pca.transform(self.optimal_genotypes)
        # df_sim = pd.DataFrame(columns=['x', 'y', 'generation', 'radius', 'type'])  # type 0-population, 1-optima
        df_pop = pd.DataFrame({'x': self.population[:, 0],
                               'y': self.population[:, 1],
                               'generation': np.zeros(self.N, dtype=int),
                               'radius': [0.005]*self.N,
                               'type': ['organism']*self.N,
                               })
        df_opt = self.optima_df(self.optimal_genotypes[0][0], self.optimal_genotypes[0][1], 0)
        df_sim = pd.concat([df_pop, df_opt], axis=0)

        for gen in range(1, generations+1):
            self.evolve(gen)
            if len(self.population) == 0:
                break

            # 3. pca population +  concat pd.DataFrames
            # pca_population = self.pca.transform(self.population)
            df = pd.DataFrame({'x': self.population[:, 0],
                               'y': self.population[:, 1],
                               'generation': np.ones(len(self.population), dtype=int) * gen,
                               'radius': [0.005] * len(self.population),
                               'type': ['organism'] * len(self.population),
                               })
            df_sim = pd.concat([df_sim, df], axis=0)

            # 4. pca optima + concat pd.DataFrames
            # pca_optima = self.pca.transform(self.optimal_genotypes)
            for i in range(len(self.optimal_genotypes)):
                x, y = self.optimal_genotypes[i]
                df = self.optima_df(x, y, gen)
                df_sim = pd.concat([df_sim, df], axis=0)

        return df_sim
  
