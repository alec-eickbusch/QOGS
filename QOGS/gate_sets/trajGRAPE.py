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


class trajGRAPE(GRAPE):
    def __init__(
        self,
        name="PI_GRAPE_control",
        jump_ops=None,
        success_op=None,
        threshold_start=1.0,
        threshold_end=2.0,
        n_traj=1000,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.jump_ops = tfq.qt2tf(jump_ops)
        self.success_op = tfq.qt2tf(success_op) # this needs to correspond to the same qubit state as the target
        self.threshold_start = threshold_start
        self.threshold_end = threshold_end
        self.n_traj = n_traj

    # @tf.function()
    def batch_state_transfer_fidelities(self, opt_params: Dict[str, tf.Variable]):
        blocks = self.gateset.batch_construct_block_operators(opt_params)
        states = tf.stack([self.initial_states] * self.parameters["N_multistart"])  # [batch/multistart index, initial state index, vector index, axis of length 1]
        states = tf.stack([states] * self.n_traj) # [n_traj, batch/multistart index, initial state index, vector index, axis of length 1]
        finals = tf.stack([self.target_states] * self.parameters["N_multistart"])
        states = tf.squeeze(states, axis=-1)
        finals = tf.squeeze(finals, axis=-1)

        R = tf.random.uniform(states.shape[0:3], minval=0, maxval=1.0, dtype=tf.float32)

        for k in tf.range(blocks.shape[0]):
            states = tf.einsum(
                "mij,tmsj->tmsi", blocks[k, ...], states
            )  # t: trajectories, m: multistart, s:multiple states
        
            # calculate norms to compare to random numbers R and decide on jumps
            # we'll do this by making a mask of the trajectories that will "jump"
            norms = tf.einsum("tmsi,tmsi->tms", tf.math.conj(states), states)

            mask = (R > tf.cast(norms, dtype=tf.float32)) # one indicates to insert a jump in that trajectory, 0 does nothing
            mask = tf.cast(mask, dtype=tf.complex64) # need to cast the mask to type complex64

            jump_op = tf.einsum("tms,ij->tmsij", mask, self.jump_ops) + tf.einsum("tms,ij->tmsij", 1 - mask, tf.eye(self.N, dtype=tf.complex64)) # this combines jump_op and identity to produce a masked jump operator on all trajs
            states = tf.einsum("tmsij,tmsj->tmsi", jump_op, states) # only support one jump op at the moment

            # now selectively re-normalize the states where we applied a jump operator
            norms = tf.math.sqrt(tf.einsum("tmsi,tmsi->tms", tf.math.conj(states), states))
            masked_norms = mask * norms + (1 - mask) * tf.ones_like(mask) # mask so that we only renormalize states where we just inserted a jump

            states = tf.einsum("tmsi,tms->tmsi", states, 1 / masked_norms)
            
            R_new = tf.random.uniform(states.shape[0:3], minval=0, maxval=1.0, dtype=tf.float32) # generate new random numbers
            R = tf.cast(mask, dtype=tf.float32) * R_new + (1 - tf.cast(mask, dtype=tf.float32)) * R # selectively update the random numbers based on the mask
        
        overlaps = tf.einsum("si...,tmsi->tms...", self.target_states_conj, states)
        fidelities = tf.math.conj(overlaps) * overlaps # this and the line above calculate trace overlap
        fidelities = tf.reduce_mean(fidelities, axis=[0, 2]) # average over trajectories and initial states. this implements "non-coherent" fidelity
        fidelities = tf.cast(tf.squeeze(fidelities), dtype=tf.float32)

        return fidelities