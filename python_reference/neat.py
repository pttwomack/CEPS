# A standard implementation of the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.
# This code is designed to be a general-purpose foundation that can be adapted
# for various tasks, including the maze-solving problem.

import random
import math
import copy


# --- Activation Functions ---
def sigmoid(x):
    """Sigmoid activation function."""
    try:
        return 1 / (1 + math.exp(-4.9 * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# --- Innovation History ---
class InnovationHistory:
    """
    Manages global innovation numbers for new genes (connections and nodes)
    to ensure consistency across the population. This is a core concept in NEAT
    that allows for meaningful crossover between different network topologies.
    """

    def __init__(self):
        self.innovation_number = 0
        self.node_innovations = {}
        self.connection_innovations = {}

    def get_connection_innovation(self, in_node_id, out_node_id):
        """
        Gets the innovation number for a new connection. If the connection
        has appeared before, it returns the existing number. Otherwise, it
        assigns a new one.
        """
        key = (in_node_id, out_node_id)
        if key in self.connection_innovations:
            return self.connection_innovations[key]
        else:
            self.innovation_number += 1
            self.connection_innovations[key] = self.innovation_number
            return self.innovation_number

    def get_node_innovation(self, connection_innovation_id):
        """
        Gets the innovation number for a new node that splits a connection.
        """
        key = connection_innovation_id
        if key in self.node_innovations:
            return self.node_innovations[key]
        else:
            self.innovation_number += 1
            self.node_innovations[key] = self.innovation_number
            return self.innovation_number


# --- Gene and Genome Definitions ---
class NodeGene:
    """Represents a node in the network."""

    def __init__(self, node_id, node_type, layer=0):
        self.id = node_id
        self.type = node_type  # 'input', 'output', or 'hidden'
        self.layer = layer  # Used to determine the feed-forward order

    def copy(self):
        return NodeGene(self.id, self.type, self.layer)


class ConnectionGene:
    """Represents a connection between two nodes."""

    def __init__(self, in_node_id, out_node_id, weight, enabled, innovation):
        self.in_node = in_node_id
        self.out_node = out_node_id
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

    def copy(self):
        return ConnectionGene(
            self.in_node, self.out_node, self.weight, self.enabled, self.innovation
        )


class Genome:
    """Represents an individual neural network."""

    def __init__(self, num_inputs, num_outputs, innovation_history):
        self.nodes = []
        self.connections = []
        self.fitness = 0
        self.species_id = None
        self.innovation_history = innovation_history

        # Create input and output nodes
        if num_inputs > 0 or num_outputs > 0:
            for i in range(num_inputs):
                self.nodes.append(NodeGene(i, "input", layer=0))
            for i in range(num_inputs, num_inputs + num_outputs):
                self.nodes.append(NodeGene(i, "output", layer=1))

            # Create initial connections
            for i in range(num_inputs):
                for o in range(num_inputs, num_inputs + num_outputs):
                    weight = random.uniform(-1, 1)
                    innov = self.innovation_history.get_connection_innovation(i, o)
                    self.connections.append(ConnectionGene(i, o, weight, True, innov))

    def copy(self):
        """Creates a copy of the genome."""
        new_genome = Genome(0, 0, self.innovation_history)
        new_genome.nodes = [n.copy() for n in self.nodes]
        new_genome.connections = [c.copy() for c in self.connections]
        new_genome.fitness = self.fitness
        new_genome.species_id = self.species_id
        return new_genome

    def get_node(self, node_id):
        """Finds a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def mutate_weights(self, prob_perturb, prob_new):
        """Mutates the weights of the connections."""
        for conn in self.connections:
            if random.random() < prob_perturb:
                conn.weight += random.uniform(-0.1, 0.1)
            elif random.random() < prob_new:
                conn.weight = random.uniform(-1, 1)

    def mutate_add_connection(self):
        """Adds a new connection between two previously unconnected nodes."""
        node1 = random.choice(self.nodes)
        node2 = random.choice(self.nodes)

        # Avoid recurrent connections to output nodes and connections to input nodes
        if node2.type == "input" or (
            node1.layer >= node2.layer
            and node1.type != "input"
            and node2.type != "output"
        ):
            return

        # Check if connection already exists
        for conn in self.connections:
            if (conn.in_node == node1.id and conn.out_node == node2.id) or (
                conn.in_node == node2.id and conn.out_node == node1.id
            ):
                return

        innov = self.innovation_history.get_connection_innovation(node1.id, node2.id)
        new_conn = ConnectionGene(
            node1.id, node2.id, random.uniform(-1, 1), True, innov
        )
        self.connections.append(new_conn)

    def mutate_add_node(self):
        """Adds a new node by splitting an existing connection."""
        if not self.connections:
            return

        conn_to_split = random.choice(self.connections)
        conn_to_split.enabled = False

        # Create new node
        new_node_id = len(self.nodes)
        in_node = self.get_node(conn_to_split.in_node)
        out_node = self.get_node(conn_to_split.out_node)
        new_node_layer = (in_node.layer + out_node.layer) / 2.0
        new_node = NodeGene(new_node_id, "hidden", layer=new_node_layer)
        self.nodes.append(new_node)

        # Create new connections
        innov1 = self.innovation_history.get_connection_innovation(
            in_node.id, new_node_id
        )
        conn1 = ConnectionGene(in_node.id, new_node_id, 1.0, True, innov1)

        innov2 = self.innovation_history.get_connection_innovation(
            new_node_id, out_node.id
        )
        conn2 = ConnectionGene(
            new_node_id, out_node.id, conn_to_split.weight, True, innov2
        )

        self.connections.append(conn1)
        self.connections.append(conn2)

    @staticmethod
    def crossover(parent1, parent2, innovation_history):
        """Performs crossover between two parent genomes."""
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1  # parent1 is the more fit parent

        child = Genome(0, 0, innovation_history)
        child.nodes = []
        child.connections = []

        # Copy nodes from the more fit parent
        for node in parent1.nodes:
            child.nodes.append(node.copy())

        # Inherit connections
        for conn1 in parent1.connections:
            matching_conn = None
            for conn2 in parent2.connections:
                if conn1.innovation == conn2.innovation:
                    matching_conn = conn2
                    break

            if matching_conn:  # Matching gene
                if random.random() < 0.5:
                    child.connections.append(conn1.copy())
                else:
                    child.connections.append(matching_conn.copy())
            else:  # Disjoint or excess gene
                child.connections.append(conn1.copy())

        return child

    @staticmethod
    def compatibility_distance(genome1, genome2, c1, c2, c3):
        """Calculates the compatibility distance between two genomes."""
        innovations1 = {conn.innovation for conn in genome1.connections}
        innovations2 = {conn.innovation for conn in genome2.connections}

        if not innovations1 or not innovations2:
            return 100  # High distance if one genome is empty

        max_innov = max(max(innovations1), max(innovations2))

        excess_genes = len(
            (innovations1 | innovations2) - (innovations1 & innovations2)
        )

        # Calculate disjoint genes
        disjoint_genes = 0
        for i in range(1, max_innov + 1):
            in1 = i in innovations1
            in2 = i in innovations2
            if (in1 and not in2 and i < max(innovations2)) or (
                not in1 and in2 and i < max(innovations1)
            ):
                disjoint_genes += 1

        # Calculate average weight difference of matching genes
        weight_diff = 0
        matching_genes = 0
        for conn1 in genome1.connections:
            for conn2 in genome2.connections:
                if conn1.innovation == conn2.innovation:
                    weight_diff += abs(conn1.weight - conn2.weight)
                    matching_genes += 1
                    break

        if matching_genes == 0:
            avg_weight_diff = 0
        else:
            avg_weight_diff = weight_diff / matching_genes

        N = max(len(genome1.connections), len(genome2.connections))
        if N < 20:
            N = 1  # Normalize for smaller genomes

        distance = (
            (c1 * excess_genes / N) + (c2 * disjoint_genes / N) + (c3 * avg_weight_diff)
        )
        return distance


# --- Neural Network Evaluation ---
class Network:
    """The phenotype network generated from a genome."""

    def __init__(self, genome):
        self.nodes = {}
        self.inputs = []
        self.outputs = []

        for ng in genome.nodes:
            self.nodes[ng.id] = {
                "value": 0.0,
                "layer": ng.layer,
                "type": ng.type,
                "out_conns": [],
            }
            if ng.type == "input":
                self.inputs.append(ng.id)
            elif ng.type == "output":
                self.outputs.append(ng.id)

        for cg in genome.connections:
            if cg.enabled:
                if cg.in_node not in self.nodes:
                    self.nodes[cg.in_node] = {
                        "value": 0.0,
                        "layer": -1,
                        "type": "hidden",
                        "out_conns": [],
                    }
                self.nodes[cg.in_node]["out_conns"].append((cg.out_node, cg.weight))

    def activate(self, input_values):
        """Activates the network and returns the output."""
        if len(input_values) != len(self.inputs):
            raise ValueError("Incorrect number of input values.")

        for node_id in self.nodes:
            self.nodes[node_id]["value"] = 0.0

        for i, val in zip(self.inputs, input_values):
            self.nodes[i]["value"] = val

        # Feed-forward activation
        layers = sorted(list(set(n["layer"] for n in self.nodes.values())))
        for layer in layers:
            for node_id, node in self.nodes.items():
                if node["layer"] == layer:
                    for out_node_id, weight in node["out_conns"]:
                        if self.nodes[out_node_id]["layer"] > layer:
                            self.nodes[out_node_id]["value"] += node["value"] * weight

            # Apply activation function to non-input nodes
            for node_id, node in self.nodes.items():
                if node["layer"] > layer and node["type"] != "input":
                    node["value"] = sigmoid(node["value"])

        return [self.nodes[o]["value"] for o in self.outputs]


# --- Main NEAT Algorithm ---
class Neat:
    """The main NEAT algorithm class."""

    def __init__(self, num_inputs, num_outputs, population_size):
        self.population_size = population_size
        self.innovation_history = InnovationHistory()
        self.population = [
            Genome(num_inputs, num_outputs, self.innovation_history)
            for _ in range(population_size)
        ]
        self.species = []

        # Hyperparameters
        self.c1 = 1.0  # Excess genes weight
        self.c2 = 1.0  # Disjoint genes weight
        self.c3 = 0.4  # Weight difference weight
        self.compatibility_threshold = 3.0
        self.mutation_rates = {"weight": 0.8, "add_conn": 0.05, "add_node": 0.03}

    def speciate(self):
        """Divides the population into species based on compatibility."""
        for s in self.species:
            s["genomes"] = []

        for genome in self.population:
            placed = False
            for s in self.species:
                dist = Genome.compatibility_distance(
                    genome, s["representative"], self.c1, self.c2, self.c3
                )
                if dist < self.compatibility_threshold:
                    s["genomes"].append(genome)
                    placed = True
                    break
            if not placed:
                self.species.append({"representative": genome, "genomes": [genome]})

        # Remove empty species and update representatives
        self.species = [s for s in self.species if s["genomes"]]
        for s in self.species:
            s["representative"] = random.choice(s["genomes"])

    def run_generation(self, evaluate_genome):
        """Runs one generation of the NEAT algorithm."""
        # Evaluate fitness
        for genome in self.population:
            genome.fitness = evaluate_genome(Network(genome))

        self.speciate()

        # Calculate adjusted fitness and offspring counts
        total_adjusted_fitness = 0
        for s in self.species:
            s["adjusted_fitness"] = sum(
                g.fitness / len(s["genomes"]) for g in s["genomes"]
            )
            total_adjusted_fitness += s["adjusted_fitness"]

        if total_adjusted_fitness == 0:  # Handle case of all zero fitness
            total_adjusted_fitness = 1

        new_population = []
        for s in self.species:
            num_offspring = round(
                (s["adjusted_fitness"] / total_adjusted_fitness) * self.population_size
            )

            # Elitism: carry over the best genome if the species is large enough
            s["genomes"].sort(key=lambda g: g.fitness, reverse=True)
            if len(s["genomes"]) >= 5:
                new_population.append(s["genomes"][0].copy())
                num_offspring -= 1

            for _ in range(num_offspring):
                if len(s["genomes"]) > 1 and random.random() < 0.75:  # Crossover
                    parent1 = random.choice(s["genomes"])
                    parent2 = random.choice(s["genomes"])
                    child = Genome.crossover(parent1, parent2, self.innovation_history)
                else:  # Asexual reproduction
                    child = random.choice(s["genomes"]).copy()

                # Mutation
                child.mutate_weights(self.mutation_rates["weight"], 0.1)
                if random.random() < self.mutation_rates["add_conn"]:
                    child.mutate_add_connection()
                if random.random() < self.mutation_rates["add_node"]:
                    child.mutate_add_node()

                new_population.append(child)

        # Refill population if needed
        while len(new_population) < self.population_size:
            new_population.append(random.choice(self.population).copy())

        self.population = new_population[: self.population_size]
