import tqdm
import numpy as np
import pandas as pd
import math
import random
from database import Graph, CelestialBody, CelestialGraph

from solver import MscColoringSolver, ALL_SOLVER
from instance_generation import create_circle, create_random_circle, create_circle_n_k, create_random_celest_graph
from solver.min_sum_simple_solver import solve_min_sum_simple_n_gon
from utils import visualize_solution_2d, visualize_graph_2d, visualize_min_sum_sol_2d, Multidict, callback_rerouter, get_lower_bounds
from angular_solver import solve
from solver.min_sum_simple_solver import solve_min_sum_simple_n_gon

def greedy_evolver_test():
    from angular_evolver import edge_order_evolver as OE
    crossover = OE.OrderUniformCrossover()
    selection = OE.uniform_wheel_selection
    mutation = OE.EdgeOrderMutation()
    solver = OE.AngularMinSumGreedySolver()
    result_sol_func = lambda x: np.array([item.solution[1] for item in x])
    fitness = OE.AngularSolverFitness(solver.__class__.__name__,
                                      remote_computation=False,
                                      fitness_goal=OE.AngularSolverFitness.MIN_SUM,
                                      solver_params={"no_sol_return": True},
                                      custom_result_func=result_sol_func)
    termination = OE.IterationTerminationConditionMet(max_iter=200)
    callback = OE.update_callback
    graph = OE.create_circle_n_k(12, 3)
    init_pop = np.zeros(200, dtype=object)
    init_pop[:] = [OE.EdgeOrderGraphGenome(graph) for i in range(200)]

    from genetic_algorithm import GeneticAlgorithm
    ga = GeneticAlgorithm(genomes=init_pop,
                          selection=selection,
                          mutation=mutation,
                          fitness=fitness,
                          crossover=crossover,
                          termCon=termination,
                          callback=callback)
    last = ga.evolve()
    return last
def get_sin(n,k):
    return np.sin((k * 2 * np.pi) / n)
def get_cos(n,k):
    return np.cos((k * 2 * np.pi) / n)
def get_distance(n,k):
    return get_sin(n,k) / (get_sin(n,k)**2 + (get_cos(n,k)-1)**2)
def create_graph():
    d1 = get_distance(9,3)
    d2 = get_distance(9,2)
    d = d1 + np.random.default_rng().random() * (d2-d1)
    body = CelestialBody((0,0), d, None)
    g = create_circle_n_k(9,4)
    graph = CelestialGraph([body], g.vertices, g.edges)
    visualize_graph_2d(graph)
    for i in range(10):
        rand_g = create_random_celest_graph(9,celest_bounds=(0.4,0.6))
        visualize_graph_2d(rand_g)
    

def main():
    solution_example_intro()
    #color_solver_check()
    #correct_msc_instances()
    #visualize_solutions_overview()
    #visualize_geo_solutions_overview()
    #visualize_solutions_overview_second_batch()
    #bnb_sols()
    #vis_full_circle_solutions()
    return

    graphs = [create_random_celest_graph(i, celest_bounds=(j*0.1+0.1, j*0.1+0.1), seed=None) for i in range(5, 8) for j in range(5)]
    
def correct_msc_instances():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from solver import MscColoringSolver
    session = get_session("angular.db")
    color_tasks = session.query(Task).filter(Task.name.like("%Color%")).all()
    msc_color_tasks = [t for t in color_tasks if session.query(Config).filter(Config.task_id == t.id, Config.param == "solver_args", Config._value_str == "{}").count() > 0]
    error_tasks = [j for t in tqdm.tqdm(color_tasks) for j in tqdm.tqdm(t.jobs) if j.solution == None or j.solution.error_message != None]
    solver = MscColoringSolver()
    for j in tqdm.tqdm(error_tasks, desc="Process error msc color instances"):
        sol = solver.solve(j.graph)
        j.solution = sol
        session.commit()

def visualize_geo_solutions_overview():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from utils.visualization import visualize_solution_scatter, VisTypes
    session = get_session("angular.db")
    min_sum_tasks = session.query(Task).filter(Task.parent_id.in_([80,107])).all()
    local_min_sum_tasks = session.query(Task).filter(Task.parent_id.in_([89,109])).all() + session.query(Task).filter(Task.id == 82).all()
    makespan_tasks = session.query(Task).filter(Task.parent_id.in_([97,111])).all()
    min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in min_sum_tasks])).all()
    local_min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in local_min_sum_tasks])).all()
    makespan_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in makespan_tasks])).all()
    #visualize_solution_scatter(min_sum_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)
    visualize_solution_scatter(local_min_sum_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)
    #visualize_solution_scatter(makespan_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)

def visualize_solutions_overview():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from instance_evolver import GraphGenome
    from utils import visualize_graph_2d, visualize_min_sum_sol_2d, visualize_solution_2d
    from matplotlib import pyplot as plt
    session = get_session("angular.db")
    from utils.visualization import visualize_solution_scatter, VisTypes
    min_sum_tasks = session.query(Task).filter(Task.parent_id == 2).all()
    local_min_sum_tasks = session.query(Task).filter(Task.parent_id == 11).all() 
    makespan_tasks = session.query(Task).filter(Task.parent_id == 19).all()
    min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in min_sum_tasks])).all()
    local_min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in local_min_sum_tasks])).all()
    makespan_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in makespan_tasks])).all()
    visualize_solution_scatter(min_sum_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)
    visualize_solution_scatter(local_min_sum_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)
    visualize_solution_scatter(makespan_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)

def visualize_solutions_overview_second_batch():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from instance_evolver import GraphGenome
    from utils import visualize_graph_2d, visualize_min_sum_sol_2d, visualize_solution_2d
    from matplotlib import pyplot as plt
    session = get_session("angular.db")
    from utils.visualization import visualize_solution_scatter, VisTypes
    min_sum_tasks = session.query(Task).filter(Task.parent_id == 45).all()
    local_min_sum_tasks = session.query(Task).filter(Task.parent_id == 40).all()
    makespan_tasks = session.query(Task).filter(Task.parent_id == 30, ~Task.name.like("%Reduced%")).all()
    min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in min_sum_tasks])).all()
    local_min_sum_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in local_min_sum_tasks])).all()
    makespan_jobs = session.query(TaskJobs).filter(TaskJobs.task_id.in_([t.id for t in makespan_tasks])).all()
    #visualize_solution_scatter(min_sum_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)
    #visualize_solution_scatter(local_min_sum_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)
    visualize_solution_scatter(makespan_jobs, "Bla", vis_type=VisTypes.All, logscale=True, loc=9)
    print("finished")

def color_solver_check():
    from database import Task, TaskJobs, Config, ConfigHolder, Graph, get_session, CelestialGraph, AngularGraphSolution
    session = get_session("angular.db")
    
    color_configs = session.query(Config).filter(Config._value_str == '"MscColoringSolver"').all()
    tasks = [c.task for c in color_configs if c.task != None]
    bad_solution_jobs = [j for t in tqdm.tqdm(tasks) for j in t.jobs if j.solution != None and len(j.solution.order) != j.graph.edge_amount]
    bad_tasks = {j.task:[] for j in bad_solution_jobs}
    for job in bad_solution_jobs:
        bad_tasks[job.task].append(job)
    
    for task in tqdm.tqdm(bad_tasks, desc="Processing bad tasks"):
        holder = ConfigHolder(task)
        solver = ALL_SOLVER[holder.solver](**holder.solver_args)
        for job in tqdm.tqdm(bad_tasks[task], desc="Processing bad jobs"):
            sol = solver.solve(job.graph)
            job.solution = sol
            session.commit()
    
def vis_full_circle_solutions():
    from instance_generation import create_circle_n_k
    graphs = [create_circle_n_k(n,n) for n in range(4, 9)]
    from solver.cp import ConstraintDependencySolver, ConstraintDependencyLocalMinSumSolver
    from solver.mip import AngularDependencySolver, AngularDependencyLocalMinSumSolver
    sols_MSSC = []
    sols_MLSSC = []

    import pickle
    try:
        with open("Circle_sols.pk", "rb") as f:
            sols_MSSC, sols_MLSSC = pickle.load(f)
    except (EOFError, FileNotFoundError):
        solver_MSSC = AngularDependencySolver()
        solver_MLSSC = AngularDependencyLocalMinSumSolver()
        for g in tqdm.tqdm(graphs):
            sols_MSSC.append(solver_MSSC.solve(g))
            sols_MLSSC.append(solver_MLSSC.solve(g))

        with open("Circle_sols.pk", "wb") as f:
            pickle.dump((sols_MSSC, sols_MLSSC), f)
    from utils import visualize_min_sum_sol_2d
    print("Min Sum Sols:")
    for s in sols_MSSC:
        visualize_min_sum_sol_2d(s)
    return
    
    

def bnb_sols():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from solver.bnb import MinSumAbstractGraphSolver, MinSumOrderSolver
    from create_gen_instances_script import create_random_instance
    import tqdm
    import pickle
    sols = None
    try:
        with open("bnb_sols.pk", "rb") as f:
            bnb1sols, bnb2sols = pickle.load(f)
    except (EOFError, FileNotFoundError):
        pass
    bnb1 = MinSumAbstractGraphSolver(time_limit=900)
    bnb2 = MinSumOrderSolver(time_limit=900)
    if not bnb1sols or not bnb2sols:
        graphs = [create_random_instance(n, edge_chance=e) for e in np.arange(0.5,1, 0.1) for n in range(5,8) for i in range(2)]    
        bnb1 = MinSumAbstractGraphSolver(time_limit=900)
        bnb2 = MinSumOrderSolver(time_limit=900)
        bnb1sols = []
        bnb2sols = []
        for g in tqdm.tqdm(graphs):
            bnb1sols.append(bnb1.solve(g))
            bnb2sols.append(bnb2.solve(g))
        sols = bnb1sols + bnb2sols
    
    for i,(s1, s2) in enumerate(zip(bnb1sols, bnb2sols)):
        s1.graph.id = i
        s1.graph_id = 1
        s2.graph.id = i
        s2.graph_id = 1
        

    with open("bnb_sols.pk", "wb") as f:
        pickle.dump((bnb1sols, bnb2sols), f)
    from utils.visualization import visualize_solution_scatter, VisTypes
    
    
    
    visualize_solution_scatter(bnb1sols+bnb2sols, "Branch and Bound runtimes", solution_type="runtime", vis_type=VisTypes.Absolute)
    visualize_solution_scatter(bnb1sols+bnb2sols, "Branch and Bound Vs Lower Bound", solution_type="min_sum", vis_type=VisTypes.VsLB)

def visualize_makespan_evolve_data():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from instance_evolver import GraphGenome
    from utils import visualize_graph_2d, visualize_min_sum_sol_2d, visualize_solution_2d
    from matplotlib import pyplot as plt
    session = get_session("angular_old.db")
    with session.no_autoflush:
        genomes = [session.query(GraphGenome)\
            .filter(GraphGenome.task_id == 5, GraphGenome.generation == i).all()
            for i in range(89)]
        
        import pandas as pa
        df = pa.DataFrame([{
            "Generation": gen.generation,
            #"GraphId": gen.graph_id,
            "Runtime": float(gen.solution.runtime)            
        } for genome_generation in genomes for gen in genome_generation])
        mean_df = df.groupby(["Generation"]).mean()
        max_df = df.groupby(["Generation"]).max()
        
        fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
        plt.suptitle("Absolute turn cost IP solver instance generation results")
        fig.set_size_inches(10,3.8)
        ax1.set_title("Mean runtime")
        ax2.set_title("Max runtime")
        ax1.set_ylabel("Runtime (s)")
        mean_df.plot(ax=ax1)
        max_df.plot(ax=ax2)
        ax1.get_legend().set_visible(False)
        ax2.get_legend().set_visible(False)
        plt.show()
        
        visualize_graph_2d(genomes[15][0].graph)
        visualize_graph_2d(genomes[25][0].graph)
        visualize_graph_2d(genomes[45][0].graph)
        visualize_graph_2d(genomes[60][0].graph)
        visualize_graph_2d(genomes[80][0].graph)
        visualize_solution_2d(genomes[15][0].solution)
        visualize_solution_2d(genomes[25][0].solution)
        visualize_solution_2d(genomes[45][0].solution)
        visualize_solution_2d(genomes[60][0].solution)
        visualize_solution_2d(genomes[80][0].solution)
        


def visualize_sols_inst_gen():
    from database import Task, TaskJobs, Config, Graph, get_session, CelestialGraph, AngularGraphSolution
    from utils import visualize_graph_2d, visualize_min_sum_sol_2d, visualize_solution_2d
    session = get_session("angular_copy.db")
    with session.no_autoflush:
        task = session.query(Task).filter(Task.task_type == "CelestialGraphInstances").one()
        task_jobs = session.query(TaskJobs).filter(TaskJobs.task == task).all()
        cel_graphs_old = session.query(CelestialGraph)\
            .filter(CelestialGraph.id.in_([job.graph_id for job in task_jobs]), CelestialGraph.vert_amount <= 8)\
            .all()
        cel_graphs = [i for i in cel_graphs_old if i.id in [44,68,81,82,96,1, 21, 12, 3]]

        min_sum_sols = [session.query(AngularGraphSolution).filter(AngularGraphSolution.graph_id == cel.id, AngularGraphSolution.is_optimal == True, AngularGraphSolution.solution_type == "min_sum").all() for cel in cel_graphs]
        local_min_sum_sols = [session.query(AngularGraphSolution).filter(AngularGraphSolution.graph_id == cel.id, AngularGraphSolution.is_optimal == True, AngularGraphSolution.solution_type == "local_min_sum").all() for cel in cel_graphs]
        makespan_sols = [session.query(AngularGraphSolution).filter(AngularGraphSolution.graph_id == cel.id, AngularGraphSolution.is_optimal == True, AngularGraphSolution.solution_type == "makespan").all() for cel in cel_graphs]
        for g, min_s, l_s, m_s in zip(cel_graphs, min_sum_sols, local_min_sum_sols, makespan_sols):
            print("Graph-id", g.id)
            visualize_graph_2d(g)

            if min_s:
                print("Minsum:")
                visualize_min_sum_sol_2d(min_s[0])
            if l_s:
                print("LocalMinSum")
                visualize_min_sum_sol_2d(l_s[0])
            if m_s:
                print("Makespan")
                visualize_solution_2d(m_s[0])
    return

def solution_example_intro():
    solver1 = ALL_SOLVER["ConstraintAbsSolver"]()
    solver2 = ALL_SOLVER["ConstraintDependencySolver"]()
    solver3 = ALL_SOLVER["ConstraintDependencyLocalMinSumSolver"]()
    s52 = create_circle_n_k(5,2)
    visualize_graph_2d(s52)
    sol1 = solver1.solve(s52)
    sol2 = solver2.solve(s52)
    sol3 = solver3.solve(s52)
    visualize_solution_2d(sol1)
    visualize_min_sum_sol_2d(sol1)
    print("makespan", sol1.makespan)
    visualize_solution_2d(sol2)
    visualize_min_sum_sol_2d(sol2)
    print("MinSum", sol2.min_sum)
    visualize_solution_2d(sol3)
    visualize_min_sum_sol_2d(sol3)
    print("LocalMinSum", sol2.local_min_sum)
    return
    

if __name__ == "__main__":
    main()



#test_ip.test_ip_solver()
#test_ip.test_ip_solver()