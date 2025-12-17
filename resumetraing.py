import neat
import pickle
import os
from main import main  # your main evaluation function
import copy

# Mainly AI
def seed_population_with_winner(config, winner, pop_size):
    """
    Create a fresh population of genomes seeded with the winner genome.
    Each genome gets the same structure as winner but fresh node IDs.
    """
    population = neat.Population(config)
    population.population.clear()

    for i in range(pop_size):
        genome = config.genome_type(i)  # create new genome with unique key
        genome.configure_new(config.genome_config)  # fresh genome
        # Copy connections/weights from winner where possible
        for conn_key, conn in winner.connections.items():
            if conn_key in genome.connections:
                genome.connections[conn_key].weight = conn.weight
        for node_key, node in winner.nodes.items():
            if node_key in genome.nodes:
                genome.nodes[node_key].bias = node.bias
        population.population[i] = genome

    population.species.speciate(config, population.population, generation=0)
    return population

def resume(conf_path: str, generations: int = 50, mode: str = "population", pop_size: int = 1000):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        conf_path
    )

    if mode == "population":
        if not os.path.exists("population.pkl"):
            raise FileNotFoundError("No saved population found.")
        with open("population.pkl", "rb") as f:
            population = pickle.load(f)
        print("Resuming from saved population...")

    elif mode == "winner":
        if not os.path.exists("winner.pkl"):
            raise FileNotFoundError("No saved winner genome found.")
        with open("winner.pkl", "rb") as f:
            winner = pickle.load(f)
        print(f"Resuming from winner genome with fresh population of {pop_size}...")
        population = seed_population_with_winner(config, winner, pop_size)

    else:
        raise ValueError(f"Unknown mode '{mode}'.")

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner_genome = population.run(main, generations)

    # Save results
    with open("trained_winner.pkl", "wb") as f:
        pickle.dump(winner_genome, f)
    with open("trained_population.pkl", "wb") as f:
        pickle.dump(population, f)

    print(f"Training resumed in mode '{mode}' with {pop_size} genomes.")
    print("Saved new winner to trained_winner.pkl")
    print("Saved new population to trained_population.pkl")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_feedforward")

    print("Choose resume mode:")
    print("1. population")
    print("2. winner")
    choice = input("Enter choice (1 or 2): ").strip()
    mode = "population" if choice != "2" else "winner"

    resume(config_path, generations=50, mode=mode, pop_size=1000)
