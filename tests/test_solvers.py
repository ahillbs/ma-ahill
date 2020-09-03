import pytest

from instance_generation import create_circle_n_k
from database import Graph

from solver.s_n_k_mirror_solver import SnkMirrorSolver

def test_snk_mirror_solver():
    circle = create_circle_n_k(9, 3)
    solver = SnkMirrorSolver()
    sol = solver.solve(circle)
    circle = create_circle_n_k(12, 3)
    sol1 = solver.solve(circle)
    circle = create_circle_n_k(14, 4)
    sol2 = solver.solve(circle)
    circle = create_circle_n_k(20, 5)
    sol3 = solver.solve(circle)
    print("\\o/")
