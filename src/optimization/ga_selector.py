"""
Genetic Algorithm (GA) Metaheuristic Feature Selection Module.

Operates on a binary chromosome representing selection/exclusion of the 1,280 EfficientNetB0 features.
Optimizes validation classification accuracy while minimizing feature dimensionality.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    MODELS_DIR,
    FIGURES_DIR,
    GA_POP_SIZE,
    GA_GENERATIONS,
    GA_CROSSOVER_RATE,
    GA_MUTATION_RATE,
    GA_ALPHA_PENALTY,
    GA_TOURNAMENT_SIZE,
    RANDOM_SEED
)
from src.optimization.fitness import evaluate_fitness


class GeneticAlgorithmFeatureSelector:
    """
    Binary Genetic Algorithm for CNN Deep Feature Selection.
    """
    def __init__(
        self,
        pop_size: int = GA_POP_SIZE,
        generations: int = GA_GENERATIONS,
        crossover_rate: float = GA_CROSSOVER_RATE,
        mutation_rate: float = GA_MUTATION_RATE,
        tournament_size: int = GA_TOURNAMENT_SIZE,
        alpha: float = GA_ALPHA_PENALTY,
        random_seed: int = RANDOM_SEED
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.alpha = alpha
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def initialize_population(self, feature_dim: int) -> np.ndarray:
        """Initializes population with binary random vectors (prob=0.5)."""
        return np.random.randint(0, 2, size=(self.pop_size, feature_dim), dtype=np.uint8)

    def tournament_selection(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """Performs tournament selection to select a parent."""
        indices = np.random.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = indices[np.argmax(fitness_scores[indices])]
        return population[best_idx].copy()

    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs uniform crossover between two parent chromosomes."""
        if np.random.rand() < self.crossover_rate:
            crossover_mask = np.random.rand(len(parent1)) < 0.5
            child1 = np.where(crossover_mask, parent1, parent2)
            child2 = np.where(crossover_mask, parent2, parent1)
            return child1.astype(np.uint8), child2.astype(np.uint8)
        return parent1.copy(), parent2.copy()

    def bit_flip_mutation(self, chromosome: np.ndarray) -> np.ndarray:
        """Applies bit-flip mutation to a chromosome."""
        mutation_mask = np.random.rand(len(chromosome)) < self.mutation_rate
        mutated = np.bitwise_xor(chromosome, mutation_mask.astype(np.uint8))
        return mutated

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """
        Runs the Genetic Algorithm search loop.
        """
        feature_dim = X_train.shape[1]
        population = self.initialize_population(feature_dim)

        best_global_fitness = -1.0
        best_global_mask = None
        best_global_acc = 0.0
        best_global_num_features = feature_dim

        history_best_fitness = []
        history_mean_fitness = []
        history_best_acc = []
        history_feature_count = []

        print(f"\nStarting Genetic Algorithm Feature Selection (Pop={self.pop_size}, Gen={self.generations})...")

        for gen in range(self.generations):
            # Evaluate population fitness in parallel using joblib
            results = Parallel(n_jobs=-1)(
                delayed(evaluate_fitness)(ind, X_train, y_train, X_val, y_val, self.alpha)
                for ind in population
            )

            fitness_scores = np.array([r[0] for r in results])
            acc_scores = np.array([r[1] for r in results])
            num_features = np.array([r[2] for r in results])

            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            gen_best_acc = acc_scores[gen_best_idx]
            gen_best_num_feat = num_features[gen_best_idx]
            gen_mean_fitness = np.mean(fitness_scores)

            history_best_fitness.append(gen_best_fitness)
            history_mean_fitness.append(gen_mean_fitness)
            history_best_acc.append(gen_best_acc)
            history_feature_count.append(gen_best_num_feat)

            if gen_best_fitness > best_global_fitness:
                best_global_fitness = gen_best_fitness
                best_global_mask = population[gen_best_idx].copy()
                best_global_acc = gen_best_acc
                best_global_num_features = gen_best_num_feat

            print(
                f"Gen {gen+1:02d}/{self.generations} | "
                f"Best Fitness: {gen_best_fitness:.4f} | "
                f"Val Acc: {gen_best_acc*100:.2f}% | "
                f"Selected Features: {gen_best_num_feat}/{feature_dim} ({(1 - gen_best_num_feat/feature_dim)*100:.1f}% reduction)"
            )

            # Generate next generation
            new_population = []
            
            # Elitism: retain global best individual
            if best_global_mask is not None:
                new_population.append(best_global_mask.copy())

            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection(population, fitness_scores)
                p2 = self.tournament_selection(population, fitness_scores)
                c1, c2 = self.uniform_crossover(p1, p2)
                c1 = self.bit_flip_mutation(c1)
                c2 = self.bit_flip_mutation(c2)
                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            population = np.array(new_population)

        # Plot evolution progress
        self._plot_ga_progress(history_best_fitness, history_mean_fitness, history_best_acc, history_feature_count, feature_dim)

        print("\n" + "=" * 60)
        print("         GENETIC ALGORITHM SELECTION RESULTS")
        print("=" * 60)
        print(f"Optimal Feature Count:    {best_global_num_features} / {feature_dim}")
        print(f"Dimensionality Reduction: {(1.0 - best_global_num_features / feature_dim) * 100:.2f}% removed")
        print(f"Best Validation Accuracy: {best_global_acc * 100:.2f}%")
        print(f"Best Fitness Score:       {best_global_fitness:.4f}")
        print("=" * 60 + "\n")

        # Save selected feature mask
        mask_path = MODELS_DIR / "ga_selected_features.npy"
        np.save(mask_path, best_global_mask)
        print(f"Saved GA feature selection mask to {mask_path}")

        return best_global_mask, history_best_fitness

    def _plot_ga_progress(self, best_fit, mean_fit, best_acc, feature_counts, total_feats):
        """Generates and saves the GA fitness and feature reduction progression curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        gens = range(1, len(best_fit) + 1)
        axes[0].plot(gens, best_fit, "b-o", label="Best Fitness")
        axes[0].plot(gens, mean_fit, "r--", label="Mean Fitness")
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Fitness Score")
        axes[0].set_title("GA Fitness Progression")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(gens, feature_counts, "g-s", label="Selected Features")
        axes[1].axhline(y=total_feats, color="gray", linestyle=":", label="Original Total (1280)")
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Feature Count")
        axes[1].set_title("Feature Subset Dimensionality Reduction")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        out_fig = FIGURES_DIR / "ga_fitness_curve.png"
        plt.savefig(out_fig, dpi=150)
        plt.close()
        print(f"Saved GA progress curve plot to {out_fig}")


def main():
    print("Starting Stage 6: Genetic Algorithm Feature Selection...")

    # Load feature arrays from models/
    X_train_path = MODELS_DIR / "X_train.npy"
    y_train_path = MODELS_DIR / "y_train.npy"
    X_val_path = MODELS_DIR / "X_val.npy"
    y_val_path = MODELS_DIR / "y_val.npy"

    if not X_train_path.exists():
        print(f"[ERROR] Extracted feature file not found: {X_train_path}. Run Stage 5 first.")
        return

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)

    print(f"Loaded features - X_train: {X_train.shape}, X_val: {X_val.shape}")

    # Run Genetic Algorithm Selector (30 generations)
    ga = GeneticAlgorithmFeatureSelector(pop_size=30, generations=30)
    best_mask, history = ga.fit(X_train, y_train, X_val, y_val)

    print("Stage 6 completed successfully!")


if __name__ == "__main__":
    main()
