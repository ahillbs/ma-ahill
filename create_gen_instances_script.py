import numpy as np
from database import Task, TaskJobs, Config, ConfigHolder, get_session, Graph
import configargparse


def _load_config():
    parser = configargparse.ArgumentParser(description="Small script to create celestial instances and group them in a task")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file (default: cel_instances_settings.yaml)',
        default="gen_instances_settings.yaml",
        is_config_file_arg=True)
    parser.add_argument('--min-n', type=int, default=6, help="Minimum amount of vertices (default: 6)")
    parser.add_argument('--max-n', type=int, default=25, help="Maximum amount of vertices (default: 25)")
    parser.add_argument('--edge-min', type=float, default=0.5, help="Minimum chance an edge will be added (default: 0.1)")
    parser.add_argument('--edge-max', type=float, default=1, help="Maximum chance an edge will be added (default: 0.8)")
    parser.add_argument('--edge-step', type=float, default=0.1, help="Chance increase for edge added per step (default: 0.1)")
    parser.add_argument('--repetitions', type=int, default=3, help="The amount of instances that will be created per vertex amount per chance step")
    parser.add_argument('--vertex-shape', type=float, default=[1,1], nargs='*', help="Shape of how the vertices are placed (elliptically). Can also contain a list of sizes which will be repeated. (default: [1,1] = cicle)")
    parser.add_argument('--url-path', type=str, default="angular.db", help="Path to sqlite database")
    parser.add_argument('--name', type=str, default="CelestialGraphInstances", help="Name of the task (default: CelestialGraphInstances)")
    parser.add_argument('--seed', type=int, default=None, help="Seed for current instance creation (default: None; will set a random seed)")
    try:
        open("gen_instances_settings.yaml")
        return parser.parse_args()
    except FileNotFoundError:
        parser._remove_action(*[action for action in parser._actions if getattr(action, "is_config_file_arg", False)])
    return parser.parse_args()
    
def main():
    config = _load_config()
    session = get_session(config.url_path)
    graphs = []
    if config.seed is None:
        seed = int(np.random.default_rng(None).integers(np.array([2**63]))[0])
        config.seed = seed
    else:
        seed = int(config.seed)
        
    gen = np.random.default_rng(seed)
    counter = 0
    assert len(config.vertex_shape) % 2 == 0
    shapes = np.array(config.vertex_shape).reshape(round(len(config.vertex_shape)/2), 2)
    l_shapes = len(shapes)
    for n in range(config.min_n, config.max_n):
        chance = config.edge_min
        while chance <= config.edge_max:
            for i in range(config.repetitions):
                graphs.append(create_random_instance(n, edge_chance=chance, seed=gen))
                counter += 1
            chance += config.edge_step
    task = Task(name=config.name, task_type="GeometricGraphInstances", status=Task.STATUS_OPTIONS.FINISHED)
    # Convert the namespace config into database config
    task_config_holder = ConfigHolder.fromNamespace(config, task=task, ignored_attributes=["url_path", "name", "config"])
    task.jobs = [TaskJobs(graph=graph, task=task) for graph in graphs]
    
    
    session.add(task)
    session.commit()
    print(counter, "instances were created. Corresponding task is", task.id)

def create_random_instance(point_amount, bounds=(0, 500), edge_chance=0.5, seed=None):
        gen = np.random.default_rng(seed)
        points = gen.integers(bounds[0], bounds[1], size=(point_amount, 2))
        ad_matrix = np.zeros((len(points), len(points)))
        tril_indices = np.tril_indices_from(ad_matrix, -1)
        tril_edges = np.array([1 if i < edge_chance else 0 for i in gen.random(size=len(tril_indices[0]))])
        while tril_edges.max() == 0:
            tril_edges = np.array([1 if i < edge_chance else 0 for i in gen.random(size=len(tril_indices[0]))])
        ad_matrix[tril_indices] = tril_edges
        ad_matrix = ad_matrix + ad_matrix.T
        graph = Graph(points, ad_matrix=ad_matrix)
        return graph

if __name__ == "__main__":
    main()