import jax.numpy as numpy
from jax import jit, vmap
import jax

from jax import tree_util

from . gradients import natural_gradients
from . gradients import cg_solve, cg_solve_parallel, cholesky_solve

try:
    import mpi4jax
    from mpi4py import MPI
except:
    pass

from . tensor_ops import unflatten_tensor_like_example, flatten_tree_into_tensor

from enum import Enum
class Solver(Enum):
    Cholesky = 0
    ConjugateGradient = 1

# Returns a CLOSURE over the actual optimizer function!
def close_over_optimizer(config, MPI_AVAILABLE):

    @jit
    def second_order_gradients(first_order, regularization_diagonal, jacobian, x_0=None):
        """Compute the second order gradients via Stochastic Reconfiguration

        Args:
            first_order (ndarray): The 1st order gradients
            regularization_diagonal (ndarray): regularization term for the S matrix
            jacobian (ndarray): Jacobian matrix, local only

        Returns:
            ndarray, same shape as first_order, which is the 2nd order grads
        """

        if config.solver == Solver.Cholesky:

            # In here, jacobian = jacobian - <jacobian>
            S_ij = numpy.matmul(jacobian.T, jacobian) / config.n_walkers

            if MPI_AVAILABLE:
                # We have to sum the matrix across ranks in distributed mode!
                S_ij, token = mpi4jax.allreduce(
                    S_ij,
                    op = MPI.SUM,
                    comm = MPI.COMM_WORLD,
                    token = None
                )

            dp_i, residual = cholesky_solve(
                S_ij                    = S_ij,
                regularization_diagonal = regularization_diagonal,
                f_i                     = first_order
            )

        elif config.solver == Solver.ConjugateGradient:

            norm =  config.n_walkers
            if MPI_AVAILABLE:
                dp_i, residual = cg_solve_parallel(
                    jacobian                = jacobian ,
                    regularization_diagonal = regularization_diagonal,
                    f_i                     = first_order,
                    x_0                     = x_0,
                    norm                    = norm
                )
            else:
                dp_i, residual = cg_solve(
                    jacobian                = jacobian,
                    regularization_diagonal = regularization_diagonal,
                    f_i                     = first_order,
                    x_0                     = x_0,
                    norm                    = norm
                )
        return dp_i, residual




    @jit
    def optimization_fn(
        dpsi_i, energy, dpsi_i_EL, 
        jacobian,
        current_w_params, opt_state):

        # These are the natural gradients:
        simple_gradients =  natural_gradients(
                        dpsi_i,
                        energy,
                        dpsi_i_EL,
                    )

        # We apply the optax optimizer to the simple gradients:
        gradients = unflatten_tensor_like_example(simple_gradients, current_w_params)
        b1      = numpy.asarray(0.9,   dtype=simple_gradients.dtype)
        b2      = numpy.asarray(0.0,   dtype=simple_gradients.dtype)
        delta   = numpy.asarray(0.001, dtype=simple_gradients.dtype)
        epsilon = numpy.asarray(0.001, dtype=simple_gradients.dtype)

        # This updates the state for RMS Prop regularization:
        # Get g2_i, apply this update, re-store it to opt_state
        g2_i = opt_state['g2_i']
        g2_i = tree_util.tree_map(
            lambda x, y : b1* x + (1 - b1) * y**2,
            g2_i, gradients
        )

        # We create a candidate opt state to replace but only if the gradients
        # are accepted by the step


        candidate_opt_state = {}

        candidate_opt_state['g2_i'] = g2_i


        # Repack the tensors back into a flat shape:
        repacked_grads, shapes, treedef = flatten_tree_into_tensor(gradients)
        g2_i_flat, shapes, treedef      = flatten_tree_into_tensor(g2_i)

        # If we're applying a 2nd order transform, make sure the diagonal
        # is properly shaped:

        regularization_diagonal = epsilon * numpy.sqrt(g2_i_flat) + 1e-4
        regularization_diagonal = regularization_diagonal.reshape(simple_gradients.shape)


        # convert the gradients to second order gradients:
        x_0 = opt_state['x_0']

        repacked_grads, residual = second_order_gradients(
            first_order             = simple_gradients,
            regularization_diagonal = regularization_diagonal,
            jacobian                = jacobian,
            x_0                     = x_0,
        )


        residual = numpy.mean(residual**2)

        candidate_opt_state['x_0'] = repacked_grads




        # Shape the gradients into params space:
        gradients = unflatten_tensor_like_example(
            repacked_grads,
            current_w_params
        )

        # Update and apply the momentum term:
        m_i  = opt_state['m_i']
        gradients  = tree_util.tree_map(
            lambda x, y : b2* x + (1 - b2) * y,
            m_i, gradients
        )
        candidate_opt_state['m_i'] = gradients




        updated_w_params = jax.tree_util.tree_map(
            lambda x, y : x - delta*y,
            current_w_params,
            gradients
        )

        # print(opt_metrics["optimizer/overlap"], flush=True)
        # Reject updates that don't overlap:
        
        return updated_w_params, candidate_opt_state, residual


    return optimization_fn
