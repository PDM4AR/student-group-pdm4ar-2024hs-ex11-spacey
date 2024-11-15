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
from pdm4ar.exercises_def.ex09 import goal
from pdm4ar.exercises_def.ex11.utils_params import PlanetParams, SatelliteParams


np.set_printoptions(precision=2)
np.set_printoptions(linewidth=200)


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

        self.eps = 1.0
        self.E = np.identity(8)
        self.eta = self.params.tr_radius
        self.max_final_time = 100  # TODO: set this smarter, maxbe max iters * time delta or something

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
        #  self.init_state = init_state
        #  self.goal_state = goal_state

        self.problem_parameters["init_state"].value = init_state.as_ndarray()
        self.problem_parameters["goal_state"].value = goal_state.as_ndarray()
        self.set_initial_guess(init_state, goal_state)

        assert self.problem_parameters["X_bar"].value is not None

        for _ in range(self.params.max_iterations):

            self._convexification()

            try:
                error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
            except cvx.SolverError:
                print(f"[!!!] SolverError: {self.params.solver} failed to solve the problem.")

            print()
            print(f"[*] SolverStatus: {self.problem.status}")
            print()

            if self._check_convergence():
                break

            self._update_trust_region()

            self.problem_parameters["X_bar"].value = self.variables["X"].value
            self.problem_parameters["U_bar"].value = self.variables["U"].value
            self.problem_parameters["p_bar"].value = self.variables["p"].value

        mycmds, mystates = self._extract_seq_from_array()

        return mycmds, mystates

    def set_initial_guess(self, init_state: SpaceshipState, goal_state: DynObstacleState) -> None:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K

        X = np.zeros((self.spaceship.n_x, K))
        U = np.zeros((self.spaceship.n_u, K))
        p = np.zeros((self.spaceship.n_p))
        p[0] = self.max_final_time

        X[:, 0] = init_state.as_ndarray()
        X[:6, -1] = goal_state.as_ndarray()
        X[-1, -1] = self.sp.m_v
        X = np.linspace(X[:, 0], X[:, -1], K, axis=1)

        self.problem_parameters["X_bar"].value = X
        self.problem_parameters["U_bar"].value = U
        self.problem_parameters["p_bar"].value = p

        print(f"\n\nInitial guess for X\n{self.problem_parameters['X_bar'].value.T}")

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.spaceship.n_x, self.params.K), name="X"),
            "U": cvx.Variable((self.spaceship.n_u, self.params.K), name="U"),
            "p": cvx.Variable(self.spaceship.n_p, name="p"),
            "nu": cvx.Variable((self.spaceship.n_x, self.params.K - 1), name="nu"),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        problem_parameters = {
            "init_state": cvx.Parameter(self.spaceship.n_x, name="init_state"),
            "goal_state": cvx.Parameter((6,), name="goal_state"),
            "X_bar": cvx.Parameter((self.spaceship.n_x, self.params.K), name="X_bar"),
            "U_bar": cvx.Parameter((self.spaceship.n_u, self.params.K), name="U_bar"),
            "p_bar": cvx.Parameter(self.spaceship.n_p, name="p_bar"),
        }
        for i in range(self.params.K - 1):
            problem_parameters["A_bar_" + str(i)] = cvx.Parameter((8, 8))
            problem_parameters["B_plus_bar_" + str(i)] = cvx.Parameter((8, 2))
            problem_parameters["B_minus_bar_" + str(i)] = cvx.Parameter((8, 2))
            problem_parameters["F_bar_" + str(i)] = cvx.Parameter((8, 1))
            problem_parameters["r_bar_" + str(i)] = cvx.Parameter((8,))

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """

        # initial state constraint
        constraints = [
            self.variables["X"][:, 0] == self.problem_parameters["init_state"],
        ]

        # initial and final inputs needs to be zero
        constraints.append(self.variables["U"][:, 0] == 0)
        constraints.append(self.variables["U"][:, -1] == 0)
        constraints.append(self.variables["U"][:, -2] == 0)

        # spaceship needs to arrive close to the goal
        constraints.append(
            cvx.norm(self.variables["X"][:6, -1] - self.problem_parameters["goal_state"]) <= self.params.stop_crit,
        )
        constraints.append(
            cvx.norm(self.variables["X"][:6, -2] - self.problem_parameters["goal_state"]) <= self.params.stop_crit,
        )

        for i in range(self.params.K - 1):

            # Control inputs, F_thrust is limited
            constraints.append(self.variables["U"][0, i] >= self.sp.thrust_limits[0])
            constraints.append(self.variables["U"][0, i] <= self.sp.thrust_limits[1])

            # Thrust angle is limited
            constraints.append(self.variables["X"][6, i] >= self.sp.delta_limits[0])
            constraints.append(self.variables["X"][6, i] <= self.sp.delta_limits[1])

            # Rate of change constraint (thrust angle change speed)
            constraints.append(self.variables["U"][1, i] >= self.sp.ddelta_limits[0])
            constraints.append(self.variables["U"][1, i] <= self.sp.ddelta_limits[1])

            # spaceship’s mass should be greater than or equal to the mass of the spaceship without fuel
            constraints.append(self.variables["X"][-1, i] >= self.sp.m_v)

            # Time constraint
            constraints.append(self.variables["p"][0] <= self.max_final_time)

        # dynamic constraints
        for i in range(self.params.K - 1):

            Ax = self.problem_parameters["A_bar_" + str(i)] @ self.variables["X"][:, i]
            Bpu = self.problem_parameters["B_plus_bar_" + str(i)] @ self.variables["U"][:, i + 1]
            Bmu = self.problem_parameters["B_minus_bar_" + str(i)] @ self.variables["U"][:, i]
            Fp = self.problem_parameters["F_bar_" + str(i)] @ self.variables["p"]
            r = self.problem_parameters["r_bar_" + str(i)]

            constraints.append(self.variables["X"][:, i + 1] == Ax + Bpu + Bmu + Fp + r + self.variables["nu"][:, i])

        # trust region constraints
        for i in range(self.params.K):
            dx = cvx.norm(self.variables["X"][:, i] - self.problem_parameters["X_bar"][:, i], p=2)
            du = cvx.norm(self.variables["U"][:, i] - self.problem_parameters["U_bar"][:, i], p=2)
            dp = cvx.norm(self.variables["p"] - self.problem_parameters["p_bar"], p=2)
            constraints.append(dx + du + dp <= self.params.tr_radius)

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # TODO: add mass objective: - self.variables["X"][-1:-1]
        objective = self.params.weight_p @ self.variables["p"] + cvx.norm(
            self.variables["nu"], p="fro"
        )  # use frobenius norm for virtual control variables

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
            self.problem_parameters["X_bar"].value,
            self.problem_parameters["U_bar"].value,
            self.problem_parameters["p_bar"].value,
        )

        for i in range(self.params.K - 1):
            self.problem_parameters["A_bar_" + str(i)].value = A_bar[:, i].reshape((8, 8))
            self.problem_parameters["B_plus_bar_" + str(i)].value = B_plus_bar[:, i].reshape((8, 2))
            self.problem_parameters["B_minus_bar_" + str(i)].value = B_minus_bar[:, i].reshape((8, 2))
            self.problem_parameters["F_bar_" + str(i)].value = F_bar[:, i].reshape((8, 1))
            self.problem_parameters["r_bar_" + str(i)].value = r_bar[:, i].reshape((8,))

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        x_dif = np.array(self.variables["X"].value - self.problem_parameters["X_bar"].value)
        p_dif = np.array(self.variables["p"].value - self.problem_parameters["p_bar"].value)

        dif = np.linalg.norm(p_dif) + np.max(np.array([np.linalg.norm(x_dif[:, i]) for i in range(self.params.K - 1)]))

        return bool(dif <= self.eps)

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        # TODO: figure out how to compute rho
        rho = 1

        if rho < self.params.rho_0:
            self.eta = max(self.params.min_tr_radius, self.eta / self.params.alpha)
            # don't change X_bar, U_bar, p_bar
        else:
            if rho < self.params.rho_1:
                self.eta = max(self.params.min_tr_radius, self.eta / self.params.alpha)
            elif rho < self.params.rho_2:
                self.eta = self.eta  # do not change tr_radius
            else:
                self.eta = min(self.params.max_tr_radius, self.eta * self.params.beta)

    def _extract_seq_from_array(self) -> tuple[DgSampledSequence[SpaceshipCommands], DgSampledSequence[SpaceshipState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        # in case my planner returns 3 numpy arrays
        U = np.array(self.variables["U"].value)
        X = np.array(self.variables["X"].value)

        F = self.variables["U"][0, :].value
        ddelta = self.variables["U"][1, :].value
        cmds_list = [SpaceshipCommands(f, dd) for f, dd in zip(F, ddelta)]

        ts = list(range(len(cmds_list)))
        mycmds = DgSampledSequence[SpaceshipCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = self.variables["X"].value.T
        states = [SpaceshipState(*v) for v in npstates]
        mystates = DgSampledSequence[SpaceshipState](timestamps=ts, values=states)

        print(f"\n\n U:\n{U.T}  \n\n X:\n{X.T} \n\n nu:\n{self.variables['nu'].value.T}")

        return mycmds, mystates
