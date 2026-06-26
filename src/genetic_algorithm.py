"""
Genetic Algorithm for Fuzzy Rule Optimisation
Author: Adham Aboulkheir | University of Essex | PhD Research
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable
import copy


@dataclass
class Individual:
    """A single individual in the GA population (a set of fuzzy rule weights)."""
    weights: np.ndarray
    fitness: float = 0.0
    
    def copy(self) -> "Individual":
        return Individual(weights=self.weights.copy(), fitness=self.fitness)


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimising fuzzy rule weights.
    
    Evolves a population of weight vectors to maximise classification
    performance (Geometric Mean) on the training set.
    """
    
    def __init__(self, n_rules: int, population_size: int = 30,
                 n_generations: int = 30, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1, mutation_sigma: float = 0.1,
                 random_state: int = 42):
        self.n_rules = n_rules
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.random_state = random_state
        self.best_individual_ = None
        self.fitness_history_ = []
    
    def _initialise_population(self) -> List[Individual]:
        """Initialise random population of weight vectors."""
        np.random.seed(self.random_state)
        population = []
        for _ in range(self.population_size):
            weights = np.random.uniform(0.1, 1.0, self.n_rules)
            population.append(Individual(weights=weights))
        return population
    
    def _evaluate(self, individual: Individual, 
                  fitness_fn: Callable[[np.ndarray], float]) -> float:
        """Evaluate fitness of an individual."""
        individual.fitness = fitness_fn(individual.weights)
        return individual.fitness
    
    def _tournament_selection(self, population: List[Individual],
                               tournament_size: int = 3) -> Individual:
        """Select individual via tournament selection."""
        competitors = np.random.choice(len(population), tournament_size, replace=False)
        winner = max(competitors, key=lambda i: population[i].fitness)
        return population[winner].copy()
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = np.random.randint(1, self.n_rules)
        child1_weights = np.concatenate([parent1.weights[:point], parent2.weights[point:]])
        child2_weights = np.concatenate([parent2.weights[:point], parent1.weights[point:]])
        
        return Individual(weights=child1_weights), Individual(weights=child2_weights)
    
    def _mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation."""
        mutated = individual.copy()
        mask = np.random.random(self.n_rules) < self.mutation_rate
        noise = np.random.normal(0, self.mutation_sigma, self.n_rules)
        mutated.weights[mask] += noise[mask]
        mutated.weights = np.clip(mutated.weights, 0.01, 2.0)
        return mutated
    
    def optimise(self, fitness_fn: Callable[[np.ndarray], float],
                 verbose: bool = True) -> np.ndarray:
        """
        Run the genetic algorithm.
        
        Parameters
        ----------
        fitness_fn : function that takes weight vector and returns fitness score
        verbose    : print progress
        
        Returns
        -------
        best_weights : optimised rule weight vector
        """
        population = self._initialise_population()
        
        # Initial evaluation
        for ind in population:
            self._evaluate(ind, fitness_fn)
        
        self.best_individual_ = max(population, key=lambda x: x.fitness).copy()
        
        for gen in range(self.n_generations):
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(self.best_individual_.copy())
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                self._evaluate(child1, fitness_fn)
                self._evaluate(child2, fitness_fn)
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            current_best = max(population, key=lambda x: x.fitness)
            
            if current_best.fitness > self.best_individual_.fitness:
                self.best_individual_ = current_best.copy()
            
            self.fitness_history_.append(self.best_individual_.fitness)
            
            if verbose and (gen + 1) % 10 == 0:
                print(f"  Generation {gen+1:3d}/{self.n_generations}: "
                      f"Best fitness = {self.best_individual_.fitness:.4f}")
        
        return self.best_individual_.weights
    
    def convergence_plot(self):
        """Plot fitness convergence over generations."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.plot(self.fitness_history_, color="#00c9b1", linewidth=2)
        ax.set_xlabel("Generation", color="white")
        ax.set_ylabel("Best Fitness (GM)", color="white")
        ax.set_title("GA Convergence", color="white")
        ax.tick_params(colors="white")
        ax.grid(alpha=0.3, color="#21262d")
        return fig


if __name__ == "__main__":
    print("Genetic Algorithm Optimisation Demo")
    print("=" * 40)
    
    np.random.seed(42)
    n_rules = 100
    
    # Simulate a fitness function (maximise weighted F1 proxy)
    true_weights = np.random.uniform(0.1, 1.0, n_rules)
    
    def fitness_fn(weights: np.ndarray) -> float:
        # Simulate classification performance based on weights
        score = 1 - np.mean((weights - true_weights) ** 2) / np.var(true_weights)
        return float(np.clip(score, 0, 1))
    
    ga = GeneticAlgorithm(n_rules=n_rules, population_size=30, n_generations=30)
    best_weights = ga.optimise(fitness_fn, verbose=True)
    
    print(f"\nFinal best fitness: {ga.best_individual_.fitness:.4f}")
    print(f"Weight range: [{best_weights.min():.3f}, {best_weights.max():.3f}]")
    print(f"Convergence: {ga.fitness_history_[0]:.4f} -> {ga.fitness_history_[-1]:.4f}")
