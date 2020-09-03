import numpy as np
import time
from . import Solver
from database import Graph, AngularGraphSolution
from utils import get_vertex_sectors, convert_graph_to_angular_abstract_graph, get_dep_graph, calculate_order

class AngularBipartiteMinSumSolver(Solver):
    solution_type = "min_sum"

    def __init__(self, **kwargs):
        super().__init__(kwargs.pop("params", None))
    
    def is_multicore(self):
        return False

    def solve(self, graph: Graph, **kwargs):
        # count how many vertices have largest angle north or south
        # Maybe choose which set get which starting position accordingly
        order = None
        start_time = time.time()
        error_message = None
        try:
            colors = kwargs.pop("colors", None)
            if colors is None:
                colors = self._calc_subsets(graph)
            # Make sure V_1 and V_2 are bipartite partitions
            assert len(colors) == graph.vert_amount
            assert np.alltrue([colors[i] != colors[j] for i,j in graph.edges])
            sectors_V = {i:
                get_vertex_sectors(
                    graph.vertices[i], 
                    graph.vertices[np.nonzero(graph.ad_matrix[i])],
                    start_north=bool(colors[i])
                    )
                for i in range(graph.vert_amount)
            }

            tripel_edges, abs_graph = convert_graph_to_angular_abstract_graph(graph, simple_graph=False, return_tripel_edges=True)
            edge_vert = {i: {} for i in range(graph.vert_amount)}
            for edge in tripel_edges:
                edge_vert[edge[0]][edge[1:]] = tripel_edges[edge]
            
            used_edges = []
            for vertex_key in sectors_V:
                sectors_info = sectors_V[vertex_key]
                if len(sectors_info) == 1:
                    continue
                non_zeros = np.nonzero(graph.ad_matrix[vertex_key])[0]
                for from_vert, to_vert, angle in sectors_info[:-1]:
                    used_edges.append(edge_vert[vertex_key][non_zeros[from_vert],non_zeros[to_vert]])
            
            dep_graph = get_dep_graph(used_edges, abs_graph)
            order = [tuple(abs_graph.vertices[i]) for i in calculate_order(dep_graph, calculate_circle_dep=True)]
        except NotImplementedError as e:
            error_message = str(e)
            raise e
        sol = AngularGraphSolution(
            graph,
            time.time() - start_time,
            self.__class__.__name__,
            self.solution_type,
            is_optimal=False,
            error_message=error_message,
            order=order
        )
        return sol

    def _calc_subsets(self, graph: Graph):
        raise NotImplementedError()