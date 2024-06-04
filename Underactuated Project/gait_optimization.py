import numpy as np
from math import cos, sin, pi, atan2, sqrt
import pydot
from functools import partial
import csv
import time

from pydrake.all import (
    AddDefaultVisualization,
    DiscreteContactApproximation,
    PidController,
    RobotDiagramBuilder,
    Simulator,
    StartMeshcat,
    namedview,
    MathematicalProgram,
    AddUnitQuaternionConstraintOnPlant,
    OrientationConstraint,
    PositionConstraint,
    RigidTransform,
    RotationMatrix,
    eq,
    SnoptSolver,
    Solve,
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    JacobianWrtVariable,
    InitializeAutoDiff,
    PiecewisePolynomial,
    MeshcatVisualizer
)
from IPython.display import SVG

from underactuated import ConfigureParser, running_as_notebook
from underactuated.multibody import MakePidStateProjectionMatrix

from spot_ik_helpers import SpotStickFigure

## Optimization for a single foot step
# We formulate a QP that computes trajectories for all joint angles, from a start to an end footstep configuration. 
# We do this for one foot step at a time, meaning the remaining three feet must stay in stance with the ground.

def autoDiffArrayEqual(a, b):
    return np.array_equal(a, b) and np.array_equal(
        ExtractGradient(a), ExtractGradient(b)
    )

def gait_optimization(plant, plant_context, spot, next_foot, foot_idx, box_height):
    q0 = plant.GetPositions(plant_context)

    body_frame = plant.GetFrameByName("body")

    PositionView = namedview(
        "Positions", plant.GetPositionNames(spot, always_add_suffix=False)
    )
    VelocityView = namedview(
        "Velocities", plant.GetVelocityNames(spot, always_add_suffix=False)
    )

    mu = 1  # rubber on rubber
    total_mass = plant.CalcTotalMass(plant_context, [spot])
    gravity = plant.gravity_field().gravity_vector()

    nq = 12
    foot_frame = [
        plant.GetFrameByName("front_left_foot_center"),
        plant.GetFrameByName("front_right_foot_center"),
        plant.GetFrameByName("rear_left_foot_center"),
        plant.GetFrameByName("rear_right_foot_center"),
    ]

    # SETUP
    T = 1.5
    N = 10
    in_stance = np.ones((4, N))
    in_stance[foot_idx, 1:-1] = 0 # foot that stepping

    # COMPUTE DESIRED Q FROM FOOT POS
    # Leg sequence: RB, RF, LF, LB
    # IK takes 3D pos with negated z
    next_foot_IK_frame = np.hstack((next_foot[:,0].reshape(4,1), np.zeros((4,1)), -1*next_foot[:,1].reshape(4,1)))
    mean_x = np.mean(next_foot[:,0])
    mean_z = np.mean(next_foot[:,1])

    # compute body orientation (psi)
    right_vec = next_foot[1,:] - next_foot[0,:] # RF - RB
    left_vec = next_foot[2,:] - next_foot[3,:] # LF - LB
    right_psi = atan2(right_vec[1], right_vec[0])
    left_psi = atan2(left_vec[1], left_vec[0])
    mean_psi = (right_psi + left_psi)/2 # average orientation of R/L feet vectors

    # foot_IK_indices = [2, 1, 3, 0] # conversion of opt indices to IK indices
    # ignore_idx = foot_IK_indices[foot_idx]
    # mean_x_stance = np.mean(np.delete(next_foot[:,0], ignore_idx))
    # mean_z_stance = np.mean(np.delete(next_foot[:,1], ignore_idx))

    sm = SpotStickFigure(x=mean_x, z=-mean_z, psi=mean_psi) # negate z for IK
    sm.set_absolute_foot_coordinates(next_foot_IK_frame)
    rb, rf, lf, lb = sm.get_leg_angles()

    q_end = plant.GetPositions(plant_context)
    q_end[4] = mean_x
    q_end[5] = mean_z
    q_end[6] = sm.y + box_height # body height
    q_end[7:10] = np.array(lf)
    q_end[10:13] = -np.array(rf)
    q_end[13:16] = np.array(lb)
    q_end[16:19] = -np.array(rb)

    # Init Prog
    prog = MathematicalProgram()

    # Time steps
    h = prog.NewContinuousVariables(N - 1, "h")
    prog.AddBoundingBoxConstraint(0.5 * T / N, 2.0 * T / N, h)
    prog.AddLinearConstraint(sum(h) >= 0.9 * T)
    prog.AddLinearConstraint(sum(h) <= 1.1 * T)

    # Create one context per time step (to maximize cache hits)
    context = [plant.CreateDefaultContext() for i in range(N)]
    ad_plant = plant.ToAutoDiffXd()

    # Joint positions and velocities
    nq = plant.num_positions()
    nv = plant.num_velocities()
    q = prog.NewContinuousVariables(nq, N, "q")
    v = prog.NewContinuousVariables(nv, N, "v")
    q_view = PositionView(q)
    v_view = VelocityView(v)
    q0_view = PositionView(q0)
    # Joint costs
    q_cost = PositionView([1] * nq)
    v_cost = VelocityView([1] * nv)

    q_cost.body_x = 10
    q_cost.body_y = 10
    q_cost.body_qx = 0
    q_cost.body_qy = 0
    q_cost.body_qz = 0
    q_cost.body_qw = 0
    q_cost.front_left_hip_x = 5
    q_cost.front_left_hip_y = 5
    q_cost.front_left_knee = 5
    q_cost.front_right_hip_x = 5
    q_cost.front_right_hip_y = 5
    q_cost.front_right_knee = 5
    q_cost.rear_left_hip_x = 5
    q_cost.rear_left_hip_y = 5
    q_cost.rear_left_knee = 5
    q_cost.rear_right_hip_x = 5
    q_cost.rear_right_hip_y = 5
    q_cost.rear_right_knee = 5
    v_cost.body_vx = 0
    v_cost.body_wx = 0
    v_cost.body_wy = 0
    v_cost.body_wz = 0
    for n in range(N):
        # Joint limits
        prog.AddBoundingBoxConstraint(
            plant.GetPositionLowerLimits(),
            plant.GetPositionUpperLimits(),
            q[:, n],
        )
        # Joint velocity limits
        prog.AddBoundingBoxConstraint(
            plant.GetVelocityLowerLimits(),
            plant.GetVelocityUpperLimits(),
            v[:, n],
        )
        # Unit quaternions
        AddUnitQuaternionConstraintOnPlant(plant, q[:, n], prog)
        # Body orientation
        prog.AddConstraint(
            OrientationConstraint(
                plant,
                body_frame,
                RotationMatrix(),
                plant.world_frame(),
                RotationMatrix(),
                0.1,
                context[n],
            ),
            q[:, n],
        )

        # Interpolate between start and end q
        q_interpol = q0 + (q_end - q0) * n / (N-1)

        # Initial guess for all joint angles is the home position
        prog.SetInitialGuess(
            q[:, n], q_interpol
        )  # Solvers get stuck if the quaternion is initialized with all zeros.

        # Running costs:
        prog.AddQuadraticErrorCost(np.diag(q_cost), q_interpol, q[:, n])
        prog.AddQuadraticErrorCost(np.diag(v_cost), [0] * nv, v[:, n])

    # Start and Final costs:
    prog.AddQuadraticErrorCost(10*np.diag(q_cost), q0, q[:, 0])
    prog.AddQuadraticErrorCost(10*np.diag(q_cost), q_end, q[:, N-1])

    # Make a new autodiff context for this constraint (to maximize cache hits)
    ad_velocity_dynamics_context = [ad_plant.CreateDefaultContext() for i in range(N)]

    def velocity_dynamics_constraint(vars, context_index):
        h, q, v, qn = np.split(vars, [1, 1 + nq, 1 + nq + nv])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(
                q,
                ad_plant.GetPositions(ad_velocity_dynamics_context[context_index]),
            ):
                ad_plant.SetPositions(ad_velocity_dynamics_context[context_index], q)
            v_from_qdot = ad_plant.MapQDotToVelocity(
                ad_velocity_dynamics_context[context_index], (qn - q) / h
            )
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            v_from_qdot = plant.MapQDotToVelocity(context[context_index], (qn - q) / h)
        return v - v_from_qdot

    for n in range(N - 1):
        prog.AddConstraint(
            partial(velocity_dynamics_constraint, context_index=n),
            lb=[0] * nv,
            ub=[0] * nv,
            vars=np.concatenate(([h[n]], q[:, n], v[:, n], q[:, n + 1])),
        )

    # Contact forces
    contact_force = [
        prog.NewContinuousVariables(3, N - 1, f"foot{foot}_contact_force")
        for foot in range(4)
    ]
    for n in range(N - 1):
        for foot in range(4):
            # Linear friction cone
            prog.AddLinearConstraint(
                contact_force[foot][0, n] <= mu * contact_force[foot][2, n]
            )
            prog.AddLinearConstraint(
                -contact_force[foot][0, n] <= mu * contact_force[foot][2, n]
            )
            prog.AddLinearConstraint(
                contact_force[foot][1, n] <= mu * contact_force[foot][2, n]
            )
            prog.AddLinearConstraint(
                -contact_force[foot][1, n] <= mu * contact_force[foot][2, n]
            )
            # normal force >=0, normal_force == 0 if not in_stance
            prog.AddBoundingBoxConstraint(
                0,
                in_stance[foot, n] * 4 * 9.81 * total_mass,
                contact_force[foot][2, n],
            )

    # Center of mass variables and constraints
    com = prog.NewContinuousVariables(3, N, "com")
    comdot = prog.NewContinuousVariables(3, N, "comdot")
    comddot = prog.NewContinuousVariables(3, N - 1, "comddot")
    # Initial and Final CoM
    prog.AddBoundingBoxConstraint(q0[4], q0[4], com[0, 0])
    prog.AddBoundingBoxConstraint(q0[5], q0[5], com[1, 0])
    prog.AddBoundingBoxConstraint(mean_x, mean_x, com[0, -1])
    prog.AddBoundingBoxConstraint(mean_z, mean_z, com[1, -1])
    # Initial CoM z vel == 0
    prog.AddBoundingBoxConstraint(0, 0, comdot[2, 0])
    # CoM height
    prog.AddBoundingBoxConstraint(0.2 + box_height, np.inf, com[2, :])

    # CoM dynamics
    for n in range(N - 1):
        # Note: The original matlab implementation used backwards Euler (here and throughout),
        # which is a little more consistent with the LCP contact models.
        prog.AddConstraint(eq(com[:, n + 1], com[:, n] + h[n] * comdot[:, n]))
        prog.AddConstraint(eq(comdot[:, n + 1], comdot[:, n] + h[n] * comddot[:, n]))
        prog.AddConstraint(
            eq(
                total_mass * comddot[:, n],
                sum(contact_force[i][:, n] for i in range(4)) + total_mass * gravity,
            )
        )

    # Angular momentum (about the center of mass)
    H = prog.NewContinuousVariables(3, N, "H")
    Hdot = prog.NewContinuousVariables(3, N - 1, "Hdot")
    prog.SetInitialGuess(H, np.zeros((3, N)))
    prog.SetInitialGuess(Hdot, np.zeros((3, N - 1)))

    # Hdot = sum_i cross(p_FootiW-com, contact_force_i)
    def angular_momentum_constraint(vars, context_index):
        q, com, Hdot, contact_force = np.split(vars, [nq, nq + 3, nq + 6])
        contact_force = contact_force.reshape(3, 4, order="F")
        if isinstance(vars[0], AutoDiffXd):
            dq = ExtractGradient(q)
            q = ExtractValue(q)
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(4):
                #x
                p_WF = plant.CalcPointsPositions(
                    context[context_index],
                    foot_frame[i],
                    [0, 0, 0],  
                    plant.world_frame(),
                )
                Jq_WF = plant.CalcJacobianTranslationalVelocity(
                    context[context_index],
                    JacobianWrtVariable.kQDot,
                    foot_frame[i],
                    [0, 0, 0],
                    plant.world_frame(),
                    plant.world_frame(),
                )

                ad_p_WF = InitializeAutoDiff(p_WF, Jq_WF @ dq)
                torque = torque + np.cross(
                    ad_p_WF.reshape(3) - com, contact_force[:, i]
                )
        else:
            if not np.array_equal(q, plant.GetPositions(context[context_index])):
                plant.SetPositions(context[context_index], q)
            torque = np.zeros(3)
            for i in range(4):
                p_WF = plant.CalcPointsPositions(
                    context[context_index],
                    foot_frame[i],
                    [0, 0, 0],
                    plant.world_frame(),
                )
                torque += np.cross(p_WF.reshape(3) - com, contact_force[:, i])
        return Hdot - torque

    for n in range(N - 1):
        prog.AddConstraint(eq(H[:, n + 1], H[:, n] + h[n] * Hdot[:, n]))
        Fn = np.concatenate([contact_force[i][:, n] for i in range(4)])
        prog.AddConstraint(
            partial(angular_momentum_constraint, context_index=n),
            lb=np.zeros(3),
            ub=np.zeros(3),
            vars=np.concatenate((q[:, n], com[:, n], Hdot[:, n], Fn)),
        )

    com_constraint_context = [ad_plant.CreateDefaultContext() for i in range(N)]

    def com_constraint(vars, context_index):
        qv, com, H = np.split(vars, [nq + nv, nq + nv + 3])
        if isinstance(vars[0], AutoDiffXd):
            if not autoDiffArrayEqual(
                qv,
                ad_plant.GetPositionsAndVelocities(
                    com_constraint_context[context_index]
                ),
            ):
                ad_plant.SetPositionsAndVelocities(
                    com_constraint_context[context_index], qv
                )
            com_q = ad_plant.CalcCenterOfMassPositionInWorld(
                com_constraint_context[context_index], [spot]
            )
            H_qv = ad_plant.CalcSpatialMomentumInWorldAboutPoint(
                com_constraint_context[context_index], [spot], com
            ).rotational()
        else:
            if not np.array_equal(
                qv, plant.GetPositionsAndVelocities(context[context_index])
            ):
                plant.SetPositionsAndVelocities(context[context_index], qv)
            com_q = plant.CalcCenterOfMassPositionInWorld(
                context[context_index], [spot]
            )
            H_qv = plant.CalcSpatialMomentumInWorldAboutPoint(
                context[context_index], [spot], com
            ).rotational()
        return np.concatenate((com_q - com, H_qv - H))

    for n in range(N):
        prog.AddConstraint(
            partial(com_constraint, context_index=n),
            lb=np.zeros(6),
            ub=np.zeros(6),
            vars=np.concatenate((q[:, n], v[:, n], com[:, n], H[:, n])),
        )

    # Kinematic constraints
    def fixed_position_constraint(vars, context_index, frame):
        q, qn = np.split(vars, [nq])
        if not np.array_equal(q, plant.GetPositions(context[context_index])):
            plant.SetPositions(context[context_index], q)
        if not np.array_equal(qn, plant.GetPositions(context[context_index + 1])):
            plant.SetPositions(context[context_index + 1], qn)
        p_WF = plant.CalcPointsPositions(
            context[context_index], frame, [0, 0, 0], plant.world_frame()
        )
        p_WF_n = plant.CalcPointsPositions(
            context[context_index + 1], frame, [0, 0, 0], plant.world_frame()
        )
        if isinstance(vars[0], AutoDiffXd):
            J_WF = plant.CalcJacobianTranslationalVelocity(
                context[context_index],
                JacobianWrtVariable.kQDot,
                frame,
                [0, 0, 0],
                plant.world_frame(),
                plant.world_frame(),
            )
            J_WF_n = plant.CalcJacobianTranslationalVelocity(
                context[context_index + 1],
                JacobianWrtVariable.kQDot,
                frame,
                [0, 0, 0],
                plant.world_frame(),
                plant.world_frame(),
            )
            return InitializeAutoDiff(
                p_WF_n - p_WF,
                J_WF_n @ ExtractGradient(qn) - J_WF @ ExtractGradient(q),
            )
        else:
            return p_WF_n - p_WF

    for i in range(4):
        for n in range(N):
            if in_stance[i, n]:
                # foot should be on the ground (world position z=box_height)
                prog.AddConstraint(
                    PositionConstraint(
                        plant,
                        plant.world_frame(),
                        [-np.inf, -np.inf, box_height],
                        [np.inf, np.inf, box_height],
                        foot_frame[i],
                        [0, 0, 0],
                        context[n],
                    ),
                    q[:, n],
                )
                if n > 0 and in_stance[i, n - 1]:
                    # feet should not move during stance.
                    prog.AddConstraint(
                        partial(
                            fixed_position_constraint,
                            context_index=n - 1,
                            frame=foot_frame[i],
                        ),
                        lb=np.zeros(3),
                        ub=np.zeros(3),
                        vars=np.concatenate((q[:, n - 1], q[:, n])),
                    )
            else:
                clearance = 0.02 + box_height
                if n == int(N/2):
                    clearance += 0.1
                prog.AddConstraint(
                    PositionConstraint(
                        plant,
                        plant.world_frame(),
                        [-np.inf, -np.inf, clearance],
                        [np.inf, np.inf, np.inf],
                        foot_frame[i],
                        [0, 0, 0],
                        context[n],
                    ),
                    q[:, n],
                )

    snopt = SnoptSolver().solver_id()
    prog.SetSolverOption(snopt, "Iterations Limits", 1e6 if running_as_notebook else 1)
    prog.SetSolverOption(
        snopt, "Major Iterations Limit", 200 if running_as_notebook else 1
    )
    prog.SetSolverOption(snopt, "Major Feasibility Tolerance", 5e-6)
    prog.SetSolverOption(snopt, "Major Optimality Tolerance", 1e-4)
    prog.SetSolverOption(snopt, "Superbasics limit", 2000)
    prog.SetSolverOption(snopt, "Linesearch tolerance", 0.9)

    result = Solve(prog)
    print(result.get_solver_id().name())
    print(result.is_success())  # We expect this to be false if iterations are limited.
    
    t_sol = np.cumsum(np.concatenate(([0], result.GetSolution(h))))
    q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
    v_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(v))

    return t_sol, q_sol, v_sol, q_end