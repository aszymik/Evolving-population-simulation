import numpy as np
import pandas as pd


def rotate(vector: np.ndarray, angle: int) -> np.ndarray:
    """Obraca wektor o dany kąt (w stopniach) i zwraca nowopowstały wektor"""
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
        self.fitness_std = fitness_std
        self.fitness_threshold = fitness_thr
        self.offspring_thresholds = np.linspace(self.fitness_threshold, 1, num=self.max_num_children+1)
        self.radii = [-2 * (self.fitness_std ** 2) * np.log(thr) if thr != 0 else 0 
                      for thr in self.offspring_thresholds]  # promienie kół związane z liczą potomstwa

    def initialize_population(self) -> np.ndarray:
        """Tworzy populację N początkowych osobników"""
        return np.array([
            self.optimal_genotypes[0] + np.random.normal(0, 0.5, self.n)
            for _ in range(self.N)
        ])

    def fitness_function(self, organism: np.ndarray) -> float:
        """Oblicza fitness w stosunku do każdego z optymalnych genotypów i zwraca największy"""
        fitness = []
        for opt in self.optimal_genotypes:
            fitness.append(np.exp(-np.linalg.norm(organism - opt)**2 / (2 * self.fitness_std**2)))
        return max(fitness)

    def mutation(self, organism: np.ndarray) -> np.ndarray:
        if np.random.uniform() < self.mutation_prob:
            mutation_index = np.random.randint(0, self.n)
            organism[mutation_index] += np.random.normal(0, self.mutation_std)
        return organism

    def selection(self) -> None:
        """Eliminuje osobniki z dopasowaniem poniżej średniej"""
        fitness_values = np.array(
            [self.fitness_function(organism) for organism in self.population])

        self.population = np.array([
            organism for organism, fitness in zip(self.population, fitness_values)
            if fitness >= self.fitness_threshold
        ])

    def reproduction(self) -> None:
        # Obliczanie dostosowania dla każdego osobnika w populacji
        fitness_scores = np.array(
            [self.fitness_function(organism) for organism in self.population])

        # Przypisanie liczby dzieci dla każdego osobnika w zależności od fitness
        offspring_groups = np.digitize(fitness_scores,
                                       self.offspring_thresholds,
                                       right=True)

        # Reprodukcja z uwzględnieniem liczby dzieci
        offspring = []
        for parent_index, num_offspring in enumerate(offspring_groups):
            for _ in range(num_offspring - 1):
                offspring.append(self.population[parent_index].copy())

        # Aktualizacja populacji
        self.population = np.array(offspring)  # osobnik żyje tylko jedno pokolenie

    def environment_change(self, generation: int) -> None:
        """Zmienia optymalne genotypy w zalezności od pokolenia"""

        if generation != 0 and generation % self.T == 0:  # kolejne optimum
            optim_idx = np.random.randint(len(self.optimal_genotypes))
            new = rotate(self.optimal_genotypes[optim_idx], angle=-self.angle)
            self.optimal_genotypes[optim_idx] = rotate(
                self.optimal_genotypes[optim_idx], angle=self.angle)
            self.optimal_genotypes.append(new)

        else:
            # Przesunięcie kazdego genotypu o env_change w jego kierunku
            self.optimal_genotypes = [
                opt + self.env_change * (opt / np.linalg.norm(opt)) for opt in self.optimal_genotypes
            ]

    def evolve(self, generation: int) -> None:
        self.population = np.array(
            [self.mutation(organism) for organism in self.population])
        self.selection()
        self.reproduction()

        # Redukcja populacji do max_N osobników
        if len(self.population) > self.max_N:
            np.random.shuffle(self.population)
            self.population = self.population[:self.max_N]

        self.environment_change(generation)
    
    def optima_df(self, x: int, y: int, gen: int) -> pd.DataFrame:
        """Tworzy DataFrame z optymalnymi genotypami, przechowując informację o promieniach kół"""
        df = pd.DataFrame(columns=['x', 'y', 'generation', 'radius', 'type'])

        for r in self.radii:
            df.loc[len(df)] = [x, y, gen, r, 'optimum']
        df['generation'] = df['generation'].astype(int)
        return df

    def simulation(self, generations: int) -> pd.DataFrame:
        """
        Symuluje ewolucję populacji przez określoną liczbę pokoleń.

        Parametry:
        generations: Liczba pokoleń do symulacji.

        Zwraca:
        pd.DataFrame: DataFrame z danymi symulacji o kolumnach:
          * x, y – współrzędne punktów,
          * generation – nr pokolenia, 
          * radius – promień punktu (wazny szczególnie w przypadku optymalnych genotypów) 
          * type:  0 – populacja, 1 – optima
        """

        # 1. Tworzenie pustego DataFrame z danymi o organizmach
        df_pop = pd.DataFrame({'x': self.population[:, 0],
                               'y': self.population[:, 1],
                               'generation': np.zeros(self.N, dtype=int),
                               'radius': [0.005]*self.N,
                               'type': ['organism']*self.N,
                               })
        
        # 2. Tworzenie DataFrame z optymalnymi genotypami
        df_opt = self.optima_df(self.optimal_genotypes[0][0], self.optimal_genotypes[0][1], 0)
        df_sim = pd.concat([df_pop, df_opt], axis=0)

        for gen in range(1, generations+1):
            self.evolve(gen)
            if len(self.population) == 0:
                break

            # 3. Dodawanie danych o organizmach do DataFrame
            df = pd.DataFrame({'x': self.population[:, 0],
                               'y': self.population[:, 1],
                               'generation': np.ones(len(self.population), dtype=int) * gen,
                               'radius': [0.005] * len(self.population),
                               'type': ['organism'] * len(self.population),
                               })
            df_sim = pd.concat([df_sim, df], axis=0)

            # 4. Dodawanie danych o optymalnych genotypach do DataFrame
            for i in range(len(self.optimal_genotypes)):
                x, y = self.optimal_genotypes[i]
                df = self.optima_df(x, y, gen)
                df_sim = pd.concat([df_sim, df], axis=0)

        return df_sim
  