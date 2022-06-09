TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # supress warnings
import h5py

import QOGS.optimizer.tf_quantum as tfq
from QOGS.optimizer.GateSynthesizer import GateSynthesizer
from QOGS.gate_sets.gate_set import GateSet
from QOGS.gate_sets import GRAPE
import qutip as qt
import datetime
import time
from .operators import HamiltonianEvolutionOperator, HamiltonianEvolutionOperatorInPlace
from typing import List, Dict
import numpy as np


class PI_GRAPE(GRAPE):
    def __init__(
        self,
        name="PI_GRAPE_control",
        jump_ops=None,
        jump_weights=0,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.jump_ops = tfq.qt2tf(jump_ops)
        self.jump_weights = jump_weights

    @tf.function()
    def batch_state_transfer_fidelities(self, opt_params: Dict[str, tf.Variable]):
        bs = self.gateset.batch_construct_block_operators(opt_params)
        inits = tf.stack([self.initial_states] * self.parameters["N_multistart"])  # [batch/multistart index, initial state index, vector index, axis of length 1]
        finals = tf.stack([self.target_states] * self.parameters["N_multistart"])
        inits = tf.squeeze(inits, axis=-1)
        finals = tf.squeeze(finals, axis=-1)

        # calculate the no-jump forward propogated states first

        forwards_arr = tf.TensorArray(tf.complex64, size=bs.shape[0], dynamic_size=False) # [time index, batch/multistart index, initial state index, vector index]
        forwards_arr = forwards_arr.write(0, inits)

        for k in tf.range(bs.shape[0]):
            inits = tf.einsum(
                "mij,msj->msi", bs[k, ...], inits
            )  # m: multistart, s:multiple states
            forwards_arr = forwards_arr.write(k, inits)
        forwards = forwards_arr.stack() # [time index, batch/multistart index, initial state index, vector index]

        # now calculate backward propogated states

        backwards_arr = tf.TensorArray(tf.complex64, size=bs.shape[0], dynamic_size=False) # [time index, batch/multistart index, initial state index, vector index]
        backwards_arr = backwards_arr.write(0, finals)

        for k in tf.range(bs.shape[0]):
            finals = tf.einsum(
                "mij,msj->msi", tf.linalg.adjoint(bs[bs.shape[0] - 1 - k, ...]), finals # run the blocks backwards, so we propagate backwards in time from the final state
            )  # m: multistart, s:multiple states
            backwards_arr = backwards_arr.write(bs.shape[0] - 1 - k, finals) # order so that the first entry of backwards is the fully back-propagated final state, else we calculate overlaps of states with equal numbers of forward/back prop
        backwards = backwards_arr.stack() # [time index, batch/multistart index, initial state index, vector index]

        one_jump_states = tf.einsum("ij,kmsj...->kmsi...", self.jump_ops, forwards) # each forward state after a jump
        one_jump_state_norms = tf.einsum("kmsi...,kmsi...->kms...", tf.math.conj(one_jump_states), one_jump_states) # renormalize after jump
        one_jump_states = tf.einsum("kmsi...,kms...->kmsi...", one_jump_states, 1 / one_jump_state_norms)
        one_jump_overlaps = self.jump_weights * tf.reduce_mean(tf.einsum("kmsi...,kmsi...->kms...", tf.math.conj(backwards), one_jump_states), axis=[2]) # calculate overlaps with single jumps inserted
        no_jump_overlaps = (1 - self.jump_weights) * tf.reduce_mean(tf.einsum("si...,msi...->ms...", self.target_states_conj, forwards[-1, :, :, :]), axis=1) # averaging over initial/final states
        
        one_jump_overlaps = tf.squeeze(one_jump_overlaps)
        no_jump_overlaps = tf.squeeze(no_jump_overlaps)
        # squeeze after reduce_mean which uses axis=1,
        # which will not exist if squeezed before for single state transfer
        fids = tf.cast(no_jump_overlaps * tf.math.conj(no_jump_overlaps), dtype=tf.float32) \
                + tf.reduce_mean(tf.cast(one_jump_overlaps * tf.math.conj(one_jump_overlaps), dtype=tf.float32), axis=0) # need reduce_mean here after mod squared to average over jump times
        return fids