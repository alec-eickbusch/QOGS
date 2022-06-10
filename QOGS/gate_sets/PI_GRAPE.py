TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # suppress warnings
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
        success_op=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.jump_ops = tfq.qt2tf(jump_ops)
        self.jump_weights = tf.cast(jump_weights, dtype=tf.complex64)
        self.success_op = tfq.qt2tf(success_op) # this needs to correspond to the same qubit state as the target

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

        # now apply jump operators to those states
        one_jump_states = tf.einsum("ij,kmsj...->kmsi...", self.jump_ops, forwards) # each forward state after a jump, may need to treat final time step carefully if jump_op != 
        one_jump_state_norms = tf.einsum("kmsi...,kmsi...->kms...", tf.math.conj(one_jump_states), one_jump_states) # calculate state norms
        one_jump_states = tf.einsum("kmsi...,kms...->kmsi...", one_jump_states, 1 / one_jump_state_norms) # renormalize after jump

        # now we need to finish propogating the forward states after the jumps

        one_jump_state_arr = tf.TensorArray(tf.complex64, size=bs.shape[0], dynamic_size=False, clear_after_read=False) # [time index, batch/multistart index, initial state index, vector index]
        one_jump_state_arr = one_jump_state_arr.write(bs.shape[0] - 1, one_jump_states[-1, ...])

        for k in tf.range(1, bs.shape[0]): # we have already propogated one time step
            one_jump_states = tf.einsum(
                "mij,kmsj->kmsi", bs[k, ...], one_jump_states # propogates all states, but we only save the finished one per loop. over-propogated states are ignored
            )  # m: multistart, s:multiple states
            one_jump_state_arr = one_jump_state_arr.write(bs.shape[0] - k - 1, one_jump_states[bs.shape[0] - k - 1, ...]) # save the finished state
        one_jump_states = one_jump_state_arr.stack()
        
        # calculate the conditional fidelity for one jump
        p_success_given_one_jump = tf.einsum("kmsi,ij,kmsj->kms", tf.math.conj(one_jump_states), self.success_op, one_jump_states)
        one_jump_overlaps = tf.reduce_mean(tf.einsum("si...,kmsi...,kms...->kms...", 
                            self.target_states_conj, one_jump_states, tf.math.sqrt(1 / p_success_given_one_jump + 2e-38)), axis=[2]) # calculate overlaps with single jumps inserted, average over start states
        one_jump_cond_fids = self.jump_weights * tf.reduce_mean(one_jump_overlaps * tf.math.conj(one_jump_overlaps), axis=[0]) # average over jump times
        one_jump_cond_fids = tf.squeeze(one_jump_cond_fids)
        
        # calculate the conditional fidelity for no jumps
        no_jump_overlaps = tf.einsum("si...,msi...->ms...", self.target_states_conj, forwards[-1, ...])
        p_success_given_no_jumps = tf.einsum("msi...,ij,msj...->ms...", tf.math.conj(forwards[-1, ...]), self.success_op, forwards[-1, ...]) + 2e-38 # calculating prob of success with no jumps
        no_jump_cond_fids = (1 - self.jump_weights) * tf.math.conj(no_jump_overlaps) * no_jump_overlaps / p_success_given_no_jumps
        no_jump_cond_fids = tf.reduce_mean(no_jump_cond_fids, axis=[1])
        no_jump_cond_fids = tf.squeeze(no_jump_cond_fids)
        # squeeze after reduce_mean which uses axis=1,
        # which will not exist if squeezed before for single state transfer
        cond_fids = tf.cast(no_jump_cond_fids, dtype=tf.float32) + tf.cast(one_jump_cond_fids, dtype=tf.float32)
        return cond_fids