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
from .operators import HamiltonianEvolutionOperator, HamiltonianEvolutionOperatorInPlace
from typing import List, Dict
import numpy as np


class PI_GRAPE(GRAPE):
    def __init__(
        self,
        name="PI_GRAPE_control",
        jump_ops=None,
        jump_prob=0,
        success_op=None,
        threshold_start=1.0,
        threshold_end=2.0,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.jump_ops = tfq.qt2tf(jump_ops)
        self.jump_prob = tf.cast(jump_prob, dtype=tf.complex64) # probability of a jump during a timestep
        self.success_op = tfq.qt2tf(success_op) # this needs to correspond to the same qubit state as the target
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end

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
        one_jump_states = tf.einsum("ij,kmsj...->kmsi...", self.jump_ops, forwards) # each forward state after a jump

        # now we need to finish propogating the forward states after the jumps

        one_jump_state_arr = tf.TensorArray(tf.complex64, size=bs.shape[0], dynamic_size=False) # [time index, batch/multistart index, initial state index, vector index]
        one_jump_state_arr = one_jump_state_arr.write(bs.shape[0] - 1, one_jump_states[-1, ...]) # add state with jump at the end

        for k in tf.range(1, bs.shape[0]): # we have already propogated one time step
            one_jump_states = tf.einsum(
                "mij,kmsj->kmsi", bs[k, ...], one_jump_states # propogates all states, but we only save the finished one per loop. over-propogated states are ignored
            )  # m: multistart, s:multiple states
            one_jump_state_arr = one_jump_state_arr.write(bs.shape[0] - k - 1, one_jump_states[bs.shape[0] - k - 1, ...]) # save the finished state
        one_jump_states = one_jump_state_arr.stack()
        
        # calculate probability of each number of jumps
        zero_jump_norms = tf.einsum("msi,msi->ms", tf.math.conj(forwards[-1, ...]), forwards[-1, ...])
        one_jump_norms = tf.einsum("kmsi...,kmsi...->ms...", tf.math.conj(one_jump_states), one_jump_states) * self.jump_prob
        
        # calculate the conditional fidelity for one jump
        p_success_given_one_jump = tf.reduce_mean(tf.einsum("kmsi...,ij,kmsj...->kms...", tf.math.conj(one_jump_states), self.success_op, one_jump_states), axis=[0])

        one_jump_overlaps = tf.einsum("si...,kmsi...->kms...", \
                            self.target_states_conj, one_jump_states) # calculate overlaps with single jumps inserted, average over start states
        one_jump_joint_fids = tf.reduce_mean(one_jump_overlaps * tf.math.conj(one_jump_overlaps), axis=[0]) # shape ms...
        
        # calculate the conditional fidelity for no jumps
        p_success_given_no_jumps = tf.einsum("msi...,ij,msj...->ms...", tf.math.conj(forwards[-1, ...]), self.success_op, forwards[-1, ...]) # calculating prob of success with no jumps

        no_jump_overlaps = tf.einsum("si...,msi...->ms...", self.target_states_conj, forwards[-1, ...])
        no_jump_joint_fids = tf.math.conj(no_jump_overlaps) * no_jump_overlaps

        success_prob = tf.einsum("ms...,ms...->ms...", p_success_given_one_jump + p_success_given_no_jumps, 1 / (zero_jump_norms + one_jump_norms))
        success_prob = tf.reduce_mean(success_prob, axis=[1])
        success_prob = tf.cast(tf.squeeze(success_prob), dtype=tf.float32)
        # squeeze after reduce_mean which uses axis=1,
        # which will not exist if squeezed before for single state transfer

        joint_prob = tf.einsum("ms...,ms...->ms...", no_jump_joint_fids + one_jump_joint_fids, 1 / (zero_jump_norms + one_jump_norms))
        joint_prob = tf.reduce_mean(joint_prob, axis=[1])
        joint_prob = tf.cast(tf.squeeze(joint_prob), dtype=tf.float32)

        max_joint_prob = tf.math.reduce_max(joint_prob) # find current maximum joint fidelity

        # once joint fidelity exceeds a certain point, transition to conditional fidelity
        # the idea is that if we maximize joint fidelity up to some threshold, we will
        # optimize the no jump fidelity, but switch to conditional for fine-tuning
        return joint_prob / (1 - (1 - success_prob) * tf.keras.activations.relu((max_joint_prob - self.threshold_start) / (self.threshold_end - self.threshold_start), max_value=1))