TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # supress warnings
import h5py

import QOGS.optimizer.tf_quantum as tfq
from QOGS.gate_sets.gate_set import GateSet
import qutip as qt
import datetime
import time
from .operators import HamiltonianEvolutionOperator, HamiltonianEvolutionOperatorInPlace
from typing import List, Dict


class GRAPE(GateSet):
    def __init__(
        self,
        H_static,
        H_control: List,
        DAC_delta_t=2,
        bandwidth=0.1, # this number is the bandwidth of the pulse as a fraction of half the sampling frequency f_s / 2. it is rounded appropriately below
        inplace=False,
        name="GRAPE_control",
        **kwargs
    ):
        super().__init__(name)
        # combine all keyword arguments
        self.parameters = {
            **self.parameters,
            **kwargs,
        }  # python 3.9: self.parameters | kwargs
        self.H_static = tfq.qt2tf(H_static)
        self.N = self.H_static.shape[0]

        self.N_drives = int(len(H_control))
        assert self.N_drives % 2 == 0

        self.H_control = []
        for k in H_control:
            self.H_control.append(tfq.qt2tf(k))

        if inplace:
            self.U = HamiltonianEvolutionOperatorInPlace(
                N=self.H_static.shape[-1], H_static=self.H_static, delta_t=DAC_delta_t
            )
        else:
            self.U = HamiltonianEvolutionOperator(
                N=self.H_static.shape[-1], H_static=self.H_static, delta_t=DAC_delta_t
            )


    @property
    def parameter_names(self):

        params = []

        for k in range(int(self.N_drives // 2)):
            params.append("I_DC" + str(k))
            params.append("I_A" + str(k))
            params.append("I_B" + str(k))
            params.append("Q_DC" + str(k))
            params.append("Q_A" + str(k))
            params.append("Q_B" + str(k))

        return params

    def randomization_ranges(self):
        # a bit of a problem here. for the Fourier ansatz, the number of parameters is decoupled from the number of blocks
        ranges = {}
        for k in range(int(self.N_drives // 2)):
            ranges["I_DC" + str(k)] = (
                -1 * self.parameters["scale"],
                1 * self.parameters["scale"],
            )
            ranges["I_A" + str(k)] = (
                -1 * self.parameters["scale"],
                1 * self.parameters["scale"],
            )
            ranges["I_B" + str(k)] = (
                -1 * self.parameters["scale"],
                1 * self.parameters["scale"],
            )
            ranges["Q_DC" + str(k)] = (
                -1 * self.parameters["scale"],
                1 * self.parameters["scale"],
            )
            ranges["Q_A" + str(k)] = (
                -1 * self.parameters["scale"],
                1 * self.parameters["scale"],
            )
            ranges["Q_B" + str(k)] = (
                -1 * self.parameters["scale"],
                1 * self.parameters["scale"],
            )

        return ranges

    @tf.function
    def batch_construct_block_operators(self, opt_vars):
        Is = []  # size of each element (N_blocks, N_batch)
        Qs = []

        for k in range(self.N_drives // 2):
            Is.append(
                tf.cast(
                        opt_vars["I_DC" + str(k)] + , dtype=tf.complex64)
            )  # will do interpolation here if need be
            Qs.append(tf.cast(opt_vars["Q" + str(k)], dtype=tf.complex64))

        # n_block, n_batch, Hilbert dimension, Hilbert dimension
        H_cs = tf.einsum("ab,cd->abcd", Is[0], self.H_control[0]) + tf.einsum(
            "ab,cd->abcd", Qs[0], self.H_control[1]
        )
        for k in range(1, self.N_drives // 2):
            H_cs += tf.einsum("ab,cd->abcd", Is[k], self.H_control[2 * k]) + tf.einsum(
                "ab,cd->abcd", Qs[k], self.H_control[2 * k + 1]
            )

        blocks = self.U(H_cs)

        return blocks
