import numpy as np
import pandas as pd
from typing import Tuple


class OptimalGenotype:
    
    def __init__(self, n, angle, env_change, vector=None):
        self.n = n
        # losujemy genotyp dla pierwszego optimum, przy kolejnych podajemy konkretny wektor
        self.genotype = vector if vector is not None else np.random.normal(0, 0.05, size=self.n)
        self.angle = angle
        self.env_change_rate = env_change
        self.history = []

    def add_organism_count(self, count) -> None:
        self.history.append(count)

    def rotate(self, env_change=None):
        """Zmienia genotyp, obracając w jednym kierunku i zwraca 
        nowy optymalny genotyp, utworzony przez obrót w przeciwną stronę"""
        x1 = self.genotype[0] * np.cos(-self.angle) - self.genotype[1] * np.sin(-self.angle)
        y1 = self.genotype[0] * np.sin(-self.angle) + self.genotype[1] * np.cos(-self.angle)

        x2 = self.genotype[0] * np.cos(self.angle) - self.genotype[1] * np.sin(self.angle)
        y2 = self.genotype[0] * np.sin(self.angle) + self.genotype[1] * np.cos(self.angle)

        self.genotype = np.array([x1, y1])
        env_change = np.random.normal(env_change[0], env_change[1]) if env_change else self.env_change_rate
        return OptimalGenotype(self.n, self.angle, env_change, np.array([x2, y2]))
    
    def env_change(self) -> None:
        self.genotype += self.env_change_rate * (self.genotype / np.linalg.norm(self.genotype))


class Population:

    def __init__(self, N, max_N, n, env_change, env_mode, T, mutation_prob, mutation_std,
                   fitness_std, fitness_thr, max_num_children, angle):
        self.N = N
        self.max_N = max_N
        self.n = n
        self.T = T   # czas, kiedy zachodzi duża zmiana w środowisku
        self.mutation_prob = mutation_prob
        self.mutation_std = mutation_std
        if env_mode == 'Global':
            self.optimal_genotypes = [OptimalGenotype(self.n, 2*np.pi*angle/360, env_change)]
        elif env_mode == 'Local':
            self.optimal_genotypes = [OptimalGenotype(self.n, 2 * np.pi * angle / 360, np.random.normal(env_change[0], env_change[1]))]
        self.population = self.initialize_population()
        self.max_num_children = max_num_children
        self.fitness_std = fitness_std
        self.fitness_threshold = fitness_thr
        self.offspring_thresholds = np.linspace(self.fitness_threshold, 1, num=self.max_num_children+1)
        self.radii = [-2 * (self.fitness_std ** 2) * np.log(thr) if thr != 0 else 0 
                      for thr in self.offspring_thresholds]  # promienie kół związane z liczą potomstwa
        self.env_mode = env_mode
        self.env_change = env_change

    def initialize_population(self) -> np.ndarray:
        """Tworzy populację N początkowych osobników"""
        return np.array([
            self.optimal_genotypes[0].genotype + np.random.normal(0, 0.5, self.n)
            for _ in range(self.N)
        ])

    def fitness_function(self, organism: np.ndarray) -> list:
        """Oblicza fitness w stosunku do każdego z optymalnych genotypów"""
        fitness_scores = []
        for optimum in self.optimal_genotypes:
            opt = optimum.genotype
            fitness_scores.append(np.exp(-np.linalg.norm(organism - opt)**2 / (2 * self.fitness_std**2)))
        return fitness_scores

    def mutation(self, organism: np.ndarray) -> np.ndarray:
        if np.random.uniform() < self.mutation_prob:
            mutation_index = np.random.randint(0, self.n)
            organism[mutation_index] += np.random.normal(0, self.mutation_std)
        return organism
    
    def selection_and_count_survivors(self) -> None:
        """Eliminuje osobniki z dopasowaniem poniżej średniej i zlicza,
         ile osobników o danym optymalnym genotypie przeżyło"""
        survivors = {i: 0 for i in range(len(self.optimal_genotypes))}
        new_population = []

        for organism in self.population:
            fitness_scores = self.fitness_function(organism)
            optimal_genotype_index = np.argmax(fitness_scores)

            if max(fitness_scores) >= self.fitness_threshold:
                survivors[optimal_genotype_index] += 1
                new_population.append(organism)

        self.population = np.array(new_population)

        # Zapisujemy liczbę osobników dla każdego genotypu
        for i, genotype in enumerate(self.optimal_genotypes):
            genotype.add_organism_count(survivors[i])

    def reproduction(self) -> None:
        # Obliczanie dostosowania dla każdego osobnika w populacji
        fitness_scores = np.array(
            [max(self.fitness_function(organism)) for organism in self.population])

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
            if self.env_mode == 'Global':
                new = self.optimal_genotypes[optim_idx].rotate()
            elif self.env_mode == 'Local':
                new = self.optimal_genotypes[optim_idx].rotate(self.env_change)
            self.optimal_genotypes.append(new)

        else:
            # Przesunięcie kazdego genotypu o env_change w jego kierunku
            for opt in self.optimal_genotypes:
                opt.env_change() 

    def evolve(self, generation: int) -> None:
        self.population = np.array(
            [self.mutation(organism) for organism in self.population])
        # self.selection()
        self.selection_and_count_survivors()
        self.reproduction()

        # Redukcja populacji do max_N osobników
        if len(self.population) > self.max_N:
            np.random.shuffle(self.population)
            self.population = self.population[:self.max_N]

        self.environment_change(generation)

    def optima_df(self, optimum, gen: int) -> pd.DataFrame:
        """Tworzy DataFrame z optymalnymi genotypami, przechowując informację o promieniach kół"""
        df = pd.DataFrame(columns=['x', 'y', 'generation', 'radius', 'type'])
        x, y = optimum.genotype[0], optimum.genotype[1]

        for r in self.radii:
            df.loc[len(df)] = [x, y, gen, r, 'optimum']
        df['generation'] = df['generation'].astype(int)
        return df 

    def simulation(self, generations: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Symuluje ewolucję populacji przez określoną liczbę pokoleń.

        Zwraca: pd.DataFrame z danymi symulacji o kolumnach:
          * x, y – współrzędne punktów,
          * generation – nr pokolenia, 
          * radius – promień punktu (wazny szczególnie w przypadku optymalnych genotypów) 
          * type:  0 – populacja, 1 – optima
        """

        # 1. Tworzenie DataFrame z danymi o organizmach
        df_pop = pd.DataFrame({'x': self.population[:, 0],
                               'y': self.population[:, 1],
                               'generation': np.zeros(self.N, dtype=int),
                               'radius': [0.005] * self.N,
                               'type': ['organism'] * self.N,
                               })
        
        # 2. Tworzenie DataFrame z optymalnymi genotypami
        # df_opt = self.optima_df(self.optimal_genotypes[0][0], self.optimal_genotypes[0][1], 0)
        df_opt = self.optima_df(self.optimal_genotypes[0], 0)
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
                df = self.optima_df(self.optimal_genotypes[i], gen)
                df_sim = pd.concat([df_sim, df], axis=0)

        # dajemy zera z przodu i dzieki temu mozemy zrobic plot
        # zwraca macierz, ktorej rzedami są historie genotypów, paddowane zerami

        # opt_data = pd.DataFrame(columns=['env_change', 'num_generations', 'extinct'])        
        # for opt in self.optimal_genotypes:
        #     if len(opt.history):
        #         num_gen = np.count_nonzero(opt.history)
        #         extinct = (opt.history[-1] == 0)
        #         opt_data.loc[len(opt_data)] = [opt.env_change_rate, num_gen, extinct]

        opt_data = np.zeros((len(self.optimal_genotypes), generations+1))
        for i, opt in enumerate(self.optimal_genotypes):
            start = generations - len(opt.history) + 1
            opt_data[i][start:] = opt.history

        return df_sim, opt_data
