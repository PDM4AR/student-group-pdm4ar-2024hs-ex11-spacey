from dataclasses import dataclass, field
from typing import Union

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.spaceship import SpaceshipCommands, SpaceshipState
from dg_commons.sim.models.spaceship_structures import (
    SpaceshipGeometry,
    SpaceshipParameters,
)

from pdm4ar.exercises.ex11.discretization import *
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant


class SpaceshipPlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    satellites: dict[PlayerName, SatelliteParams]
    spaceship: SpaceshipDyn
    sg: SpaceshipGeometry
    sp: SpaceshipParameters
    params: SolverParameters

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        satellites: dict[PlayerName, SatelliteParams],
        sg: SpaceshipGeometry,
        sp: SpaceshipParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.satellites = satellites
        self.sg = sg
        self.sp = sp

        # Solver Parameters
        self.params = SolverParameters()

        # Spaceship Dynamics
        self.spaceship = SpaceshipDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Spaceship, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.spaceship, self.params.K, self.params.N_sub)

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        self.eps = 1.0

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SpaceshipState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = goal_state

        while True:

            self._convexification()

            try:
                error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")

            if self._check_convergence():
                break

            self._update_trust_region()

        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array()

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        U = np.zeros((self.spaceship.n_u, K))
        p = np.zeros((self.spaceship.n_p))

        return X, U, p

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K)),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K)),
            "p": cvx.Variable(self.spaceship.n_p),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x),
            "goal_state": cvx.Parameter((6,)),
        }
        for i in range(self.params.K - 1):
            problem_parameters["A_" + str(i)] = cvx.Parameter((64,))
            problem_parameters["B_plus_" + str(i)] = cvx.Parameter((16,))
            problem_parameters["r_bar_" + str(i)] = cvx.Parameter((8,))

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        constraints = [
            self.variables["X"][:, 0] == self.problem_parameters["init_state"],
            cvx.norm(self.variables["X"][:6, -1] - self.problem_parameters["goal_state"]) <= self.params.stop_crit,
        ]

        for i in range(self.params.K - 1):
            # mass constraint
            constraints.append(self.variables["X"][-1, i] >= self.sp.m_v)
            # F thrust constraint
            constraints.append(self.variables["U"][0, i] >= self.sp.thrust_limits[0])
            constraints.append(self.variables["U"][0, i] <= self.sp.thrust_limits[1])
            # Thrust angle constraint
            constraints.append(self.variables["X"][6, i] >= self.sp.delta_limits[0])
            constraints.append(self.variables["X"][6, i] <= self.sp.delta_limits[0])
            # Time constraint
            constraints.append(self.variables["p"][0] <= self.params.max_iterations)
            # Rate of change constraint
            constraints.append(self.variables["U"][1, i] >= self.sp.ddelta_limits[0])
            constraints.append(self.variables["U"][1, i] <= self.sp.ddelta_limits[1])

        # dynamic constraints
        for i in range(self.params.K):
            continue

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Example objective
        # - self.variables["X"][-1:-1]
        objective = self.params.weight_p @ self.variables["p"]

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )

        print(A_bar.shape)
        print(B_plus_bar.shape)
        print(F_bar.shape)
        print(r_bar.shape)

        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        for i in range(self.params.K - 1):
            pass
            # self.problem_parameters["A_" + str(i)].value = A_bar[:, i]
            # self.problem_parameters["U_" + str(i)].value = cvx.Parameter((2, 2))
            # self.problem_parameters["p_" + str(i)].value = cvx.Parameter((1,))
        self.problem_parameters["goal_state"].value = self.goal_state.as_ndarray()

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        dynamic_dif = self.variables["X"] - self.X_bar
        dif = np.linalg.norm(self.p_bar - self.variables["p"]) + np.max(
            np.vstack([np.linalg.norm(dynamic_dif[:, i]) for i in range(self.params.K - 1)])
        )

        return bool(np.array(dif) <= self.eps)

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        pass

    def _extract_seq_from_array(self) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F = np.array(self.variables["U"][0, :])
        ddelta = np.array(self.variables["U"][1, :])
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.array(self.variables["X"]).T
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)
        return mycmds, mystates
