from typing import List
import time
import math
import numpy as np
from os import cpu_count
import concurrent.futures
from genetic_algorithm import GeneticAlgorithm, update_callback, uniform_wheel_selection
from genetic_algorithm.termination_condition import TimeConstraintTermination, IterationTerminationConditionMet, NoImprovementsTermination, TerminationCombination
from angular_evolver.edge_order_evolver import EdgeOrderGraphGenome, EdgeOrderMutation, OrderUniformCrossover
from solver import AngularMinSumGreedySolver, AngularMinSumGreedyConnectedSolver, AngularLocalMinSumGreedySolver, AngularMakespanGreedySolver
from solver import Solver
from database import Graph, AngularGraphSolution
from utils import calculate_times, get_graph_angles

def _inner_fitness_min_sum(angles, edges, orders):
    results = np.zeros(len(orders), dtype=float)
    for i, order in enumerate(orders):
        arg_sort = np.argsort(order)
        times, head_sum = calculate_times(edges[arg_sort], angles=angles, return_angle_sum=True)
        results[i] = -sum(head_sum)
    return results
def _inner_fitness_local_min_sum(angles, edges, orders):
    results = np.zeros(len(orders), dtype=float)
    for i, order in enumerate(orders):
        arg_sort = np.argsort(order)
        times, head_sum = calculate_times(edges[arg_sort], angles=angles, return_angle_sum=True)
        results[i] = -max(head_sum)
    return results
def _inner_fitness_makespan(angles, edges, orders):
    results = np.zeros(len(orders), dtype=float)
    for i, order in enumerate(orders):
        arg_sort = np.argsort(order)
        times = calculate_times(edges[arg_sort], angles=angles)
        results[i] = -max([times[key] for key in times])
    return results

def _greedy_order(orders, graph: Graph, solver: Solver):
    results = []
    for orig_order in orders:
        arg_ordered = np.argsort(orig_order)
        ordered = orig_order[arg_ordered]
        inner_graph = Graph(graph.vertices, graph.edges[arg_ordered])
        order, cost = solver.solve(inner_graph, no_sol_return=True)
        edge_dict = {(o[0], o[1]): i for i, o in enumerate(graph.edges)}
        order_translate = {edge_dict[(e1, e2)]: i for i, (e1, e2) in enumerate(order)}
        new_order = np.array([ordered[order_translate[key]] for key in range(len(orig_order))])
        times, heads = calculate_times(graph.edges[np.argsort(new_order)], graph, return_angle_sum=True)
        try:
            assert (graph.edges[np.argsort(new_order)] == order).all(), "Not all edges are sorted the same"
        except AssertionError as ar:
            print(ar)
        results.append((new_order, -cost))
    return results

class AngularGeneticMinSumSolver(Solver):
    solution_type = "min_sum"

    def is_multicore(self):
        return True
    
    def __init__(self, pop_size=200, time_limit=900, max_iterations=300, no_improvement_threshhold=60, mutation_chance=0.03, mutation_chance_gene=0.03, mutation_chance_greedy=0.6, elitism=0.03, greedy_start=True):
        self.max_time = time_limit
        self.time_limit = self.max_time
        self.termination_condition = TerminationCombination(
            [TimeConstraintTermination(time_limit),
             IterationTerminationConditionMet(max_iter=max_iterations),
             NoImprovementsTermination(max_iter=no_improvement_threshhold)
            ])
        self.mutation = EdgeOrderMutation()
        self.mutation_chance_genome = mutation_chance
        self.mutation_chance_gene = mutation_chance_gene
        self.mutation_chance_greedy = mutation_chance_greedy
        self.elitism = elitism
        self.crossover = OrderUniformCrossover()
        self.selection = uniform_wheel_selection
        self.greedy_start = greedy_start
        self.pop_size = pop_size
        self.callback = update_callback
        self.executor = None
        self.sub_solvers = [AngularMinSumGreedyConnectedSolver(), AngularMinSumGreedySolver()]
        self.graph = None
        self._inner_fitness = _inner_fitness_min_sum
        

    def _mutation(self, genomes: np.array, genome_mutation_chance=0.03, dna_mutation_chance=0.03):
        #ToDo: mutation with other metaheuristic instead of greedy
        genome_size = genomes.shape[0]
        elite = math.ceil(genome_size * self.elitism) if self.elitism < 1 else self.elitism
        # repair genomes with two or more same numbers
        for genome in genomes:
            uniques, index, counts = np.unique(genome.order_numbers, return_counts=True, return_index=True)
            if len(uniques) != len(genome):
                where = np.where(counts > 1)[0]
                for w in where:
                    wrong_index = index[w]
                    number = uniques[w]
                    if w > 0:
                        number_before = uniques[w-1]
                    else:
                        number_before = 0
                    genome.order_numbers[wrong_index] = np.random.randint(number_before, number)
                #print("halp behaind")
        if np.random.random() > self.mutation_chance_greedy:
            dna_size = len(genomes[0])
            # random choice of chosen genome via random indices
            genomes_chance = np.random.random(genome_size - elite)
            mutation_indices = np.where(genomes_chance < genome_mutation_chance)[0]
            mutation_indices += elite
            futures = [
                self.executor.submit(
                    _greedy_order,
                    [genomes[mutation_index].order_numbers], self.graph,
                    self.sub_solvers[np.random.randint(0, len(self.sub_solvers))]
                    #self._greedy_order, genomes[mutation_index],\
                    #self.sub_solvers[np.random.randint(0, len(self.sub_solvers))]
                ) for mutation_index in mutation_indices
            ]
            concurrent.futures.wait(futures, timeout=20)
            results = [result for future in futures for result in future.result()]
            for genome, result in zip(genomes[mutation_indices], results):
                order_numbers, cost = result
                genome.order_numbers = order_numbers
                genome.solution = cost
                genome.greedy = True

        else:
            self.mutation(genomes[elite:], genome_mutation_chance, dna_mutation_chance)
    
    def solve(self, graph: Graph, **kwargs):
        if graph.costs is None:
            graph.costs = get_graph_angles(graph)
        error_message = None
        self.time_limit = kwargs.pop("time_limit", self.max_time)
        self.graph = graph
        try:
            self.start_time = time.time()
            genomes = np.zeros(self.pop_size, dtype=object)
            genomes[:] = [EdgeOrderGraphGenome(graph, skip_graph_calc=True) for i in range(self.pop_size)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                self.executor = executor
                if self.greedy_start:
                    slicings = self._get_slicing(genomes)
                    futures = [
                        executor.submit(
                            _greedy_order,
                            [genome.order_numbers for genome in genomes[slicing]], self.graph,
                            self.sub_solvers[np.random.randint(0, len(self.sub_solvers))]
                            #self._greedy_order, genomes[slicing],\
                            #self.sub_solvers[np.random.randint(0, len(self.sub_solvers))]
                        ) for slicing in slicings
                    ]
                    timeout_time = self.time_limit - (time.time() - self.start_time)
                    concurrent.futures.wait(futures, timeout=timeout_time)
                    for future in futures:
                        future.exception()
            
                # Start solver
                genetic_algorithm = GeneticAlgorithm(
                    genomes=genomes,
                    selection=self.selection,
                    mutation=self._mutation,
                    fitness=self._fitness,
                    crossover=self.crossover,
                    callback=self.callback,
                    termCon=self.termination_condition,
                    elitism=self.elitism,
                    mutationChance=self.mutation_chance_genome,
                    mutationChanceGene=self.mutation_chance_gene
                )
                last_gen = genetic_algorithm.evolve()
                
        except concurrent.futures.TimeoutError as time_error:
            error_message = str(time_error)
            last_gen = genetic_algorithm.genomes
        finally:
            self.executor = None
            self.graph = None

        max_index = np.argmax([gen.solution for gen in last_gen])
        arg_sorted = np.argsort(last_gen[max_index].order_numbers)
        order = graph.edges[arg_sorted].tolist()
        sol = AngularGraphSolution(
            graph,
            time.time() - self.start_time,
            self.__class__.__name__,
            self.solution_type,
            is_optimal=False,
            order=order,
            error_message=error_message
        )
        return sol

    def _fitness(self, genomes: List[EdgeOrderGraphGenome]):
        to_solve_len = sum([1 for genome in genomes if genome.solution is None])
        to_solve = np.zeros(to_solve_len, dtype=object)
        to_solve[:] = [genome for genome in genomes if genome.solution is None]

        if len(to_solve) > 0:
            slicings = self._get_slicing(to_solve)
            
            futures = [
                self.executor.submit(
                    self._inner_fitness, self.graph.costs, self.graph.edges.copy(),
                    [genome.order_numbers for genome in to_solve[slicing]]
                ) for slicing in slicings
            ]
            timeout_time = self.time_limit - (time.time() - self.start_time)
            concurrent.futures.wait(futures, timeout=timeout_time)
            solve_results = np.array([result for future in futures for result in future.result()])

            for genome, result in zip(to_solve, solve_results):
                genome.solution = result
        results = np.array([genome.solution for genome in genomes])
        return results

    def _get_slicing(self, unsolved, concurrent_jobs=(3*cpu_count())):
        slice_size = round(len(unsolved) / concurrent_jobs)
        slice_amount = concurrent_jobs

        for i in range(slice_amount-1):
            yield slice(i*slice_size, (i+1)*slice_size)
        yield slice((slice_amount-1)*slice_size, len(unsolved))


class AngularGeneticLocalMinSumSolver(AngularGeneticMinSumSolver):
    solution_type = "local_min_sum"
    def __init__(self, pop_size=200, time_constr=900, max_iterations=300, no_improvement_threshhold=60, mutation_chance=0.03, mutation_chance_gene=0.03, mutation_chance_greedy=0.6, elitism=0.03, greedy_start=True):
        super().__init__(pop_size, time_constr, max_iterations, no_improvement_threshhold, mutation_chance, mutation_chance_gene, mutation_chance_greedy, elitism, greedy_start)
        self._inner_fitness = _inner_fitness_local_min_sum
        self.sub_solvers = [AngularLocalMinSumGreedySolver()]

class AngularGeneticMakespanSolver(AngularGeneticMinSumSolver):
    solution_type = "makespan"
    def __init__(self, pop_size=200, time_constr=900, max_iterations=300, no_improvement_threshhold=60, mutation_chance=0.03, mutation_chance_gene=0.03, mutation_chance_greedy=0.6, elitism=0.03, greedy_start=True):
        super().__init__(pop_size, time_constr, max_iterations, no_improvement_threshhold, mutation_chance, mutation_chance_gene, mutation_chance_greedy, elitism, greedy_start)
        self._inner_fitness = _inner_fitness_makespan
        self.sub_solvers = [AngularMakespanGreedySolver()]