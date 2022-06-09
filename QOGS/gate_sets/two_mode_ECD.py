TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # suppress warnings
import h5py

import QOGS.optimizer.tf_quantum as tfq
from QOGS.gate_sets.gate_set import GateSet
import qutip as qt
import datetime
import time
from .operators import DisplacementOperator

# Two mode ECD assumes two cavities
# coupled to an ancilla qubit.
class TwoModeECDGateSet(GateSet):
    def __init__(self, N_blocks=20, name="ECD_control", **kwargs):
        super().__init__(name)
        # combine all keyword arguments
        self.parameters = {
            **self.parameters,
            **kwargs,
        }  # python 3.9: self.parameters | kwargs
        self.disp_op_a = lambda alpha: tfq.two_mode_op(
            DisplacementOperator(self.parameters["N_cav"])(alpha), 0
        )
        self.disp_op_b = lambda alpha: tfq.two_mode_op(
            DisplacementOperator(self.parameters["N_cav"])(alpha), 1
        )

    @property
    def parameter_names(self):
        return [
            "betas_rho_a",
            "betas_rho_b",
            "betas_angle_a",
            "betas_angle_b",
            "phis",
            "etas",
            "thetas",
        ]

    def randomization_ranges(self):
        return {
            "betas_rho_a": (
                -1 * self.parameters["beta_scale"],
                1 * self.parameters["beta_scale"],
            ),
            "betas_rho_b": (
                -1 * self.parameters["beta_scale"],
                1 * self.parameters["beta_scale"],
            ),
            "betas_angle_a": (-np.pi, np.pi),
            "betas_angle_b": (-np.pi, np.pi),
            "phis": (-np.pi, np.pi),
            "etas": (-np.pi, np.pi),
            "thetas": (-np.pi, np.pi),
        }

    @tf.function
    def batch_construct_block_operators(self, opt_vars):
        betas_rho_a = opt_vars["betas_rho_a"]
        betas_rho_b = opt_vars["betas_rho_b"]
        betas_angle_a = opt_vars["betas_angle_a"]
        betas_angle_b = opt_vars["betas_angle_b"]
        phis = opt_vars["phis"]
        thetas = opt_vars["thetas"]
        etas = opt_vars["etas"]

        # conditional displacements
        Bs_a = (
            tf.cast(betas_rho_a, dtype=tf.complex64)
            / tf.constant(2, dtype=tf.complex64)
            * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(betas_angle_a, dtype=tf.complex64)
            )
        )

        Bs_b = (
            tf.cast(betas_rho_b, dtype=tf.complex64)
            / tf.constant(2, dtype=tf.complex64)
            * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(betas_angle_b, dtype=tf.complex64)
            )
        )

        # TODO: remove this old stuff: related to displacements in ECD gate set.
        # D = tf.zeros((1, betas_rho_a.shape[1]))
        # ds_end = self.disp_op_a(D)
        ds_end = tf.eye(
            self.parameters["N_cav"] * self.parameters["N_cav"],
            self.parameters["N_cav"] * self.parameters["N_cav"],
            batch_shape=(thetas.shape[0], thetas.shape[1]),
            dtype=tf.complex64,
        )
        # ds_end = tf.eye(self.parameters['N_cav'], batch_shape=) # figure this out; faster

        ds_g_a = self.disp_op_a(Bs_a)
        ds_e_a = tf.linalg.adjoint(ds_g_a)
        ds_g_b = self.disp_op_a(Bs_b)
        ds_e_b = tf.linalg.adjoint(ds_g_b)

        Phis = phis - tf.constant(np.pi, dtype=tf.float32) / tf.constant(
            2, dtype=tf.float32
        )
        Thetas = thetas / tf.constant(2, dtype=tf.float32)
        Phis = tf.cast(
            tf.reshape(Phis, [Phis.shape[0], Phis.shape[1], 1, 1]), dtype=tf.complex64
        )
        etas = tf.cast(
            tf.reshape(etas, [etas.shape[0], etas.shape[1], 1, 1]), dtype=tf.complex64
        )
        Thetas = tf.cast(
            tf.reshape(Thetas, [Thetas.shape[0], Thetas.shape[1], 1, 1]),
            dtype=tf.complex64,
        )

        exp = tf.math.exp(tf.constant(1j, dtype=tf.complex64) * Phis)
        im = tf.constant(1j, dtype=tf.complex64)
        exp_dag = tf.linalg.adjoint(exp)
        cos = tf.math.cos(Thetas)
        sin = tf.math.sin(Thetas)
        cos_e = tf.math.cos(etas)
        sin_e = tf.math.sin(etas)

        # constructing the blocks of the matrix
        ul = (cos + im * sin * cos_e) * ds_g_a * ds_g_b
        ll = exp * sin * sin_e * ds_e_a * ds_e_b
        ur = (
            tf.constant(-1, dtype=tf.complex64)
            * exp_dag
            * sin
            * sin_e
            * ds_g_a
            * ds_g_b
        )
        lr = (cos - im * sin * cos_e) * ds_e_a * ds_e_b

        # without pi pulse, block matrix is:
        # (ul, ur)
        # (ll, lr)
        # however, with pi pulse included:
        # (ll, lr)
        # (ul, ur)
        # pi pulse also adds -i phase, however don't need to trck it unless using multiple oscillators.a
        # append a final block matrix with a single displacement in each quadrant
        blocks = tf.concat(
            [
                -1j * tf.concat([tf.concat([ll, lr], 3), tf.concat([ul, ur], 3)], 2),
                tf.concat(
                    [
                        tf.concat([ds_end, tf.zeros_like(ds_end)], 3),
                        tf.concat([tf.zeros_like(ds_end), ds_end], 3),
                    ],
                    2,
                ),
            ],
            0,
        )
        return blocks

    def preprocess_params_before_saving(self, opt_params, *args):
        processed_params = {}
        processed_params["betas_a"] = tf.Variable(
            tf.cast(opt_params["betas_rho_a"], dtype=tf.complex64)
            * tf.math.exp(1j * tf.cast(opt_params["betas_angle_a"], dtype=tf.complex64)),
            name="betas",
            dtype=tf.complex64,
        )
        processed_params["betas_b"] = tf.Variable(
            tf.cast(opt_params["betas_rho_b"], dtype=tf.complex64)
            * tf.math.exp(1j * tf.cast(opt_params["betas_angle_b"], dtype=tf.complex64)),
            name="betas",
            dtype=tf.complex64,
        )
        processed_params["phis"] = (opt_params["phis"] + np.pi) % (2 * np.pi) - np.pi
        processed_params["etas"] = (opt_params["etas"] + np.pi) % (2 * np.pi) - np.pi
        processed_params["thetas"] = (opt_params["thetas"] + np.pi) % (
            2 * np.pi
        ) - np.pi

        return processed_params
