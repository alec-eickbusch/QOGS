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
        self.use_conditional_fid = False
        if success_op is not None:
            self.success_op = tfq.qt2tf(success_op) # this needs to correspond to the same qubit state as the target
            self.threshold_start = threshold_start
            self.threshold_end = threshold_end
            self.use_conditional_fid = True
        self.n_traj = n_traj

    # @tf.function()
    def batch_state_transfer_fidelities(self, opt_params: Dict[str, tf.Variable]):
        blocks = self.gateset.batch_construct_block_operators(opt_params)
        states = tf.stack([self.initial_states] * self.parameters["N_multistart"])
        jump_states = tf.stack([states] * self.n_traj) # [n_traj, batch/multistart index, initial state index, vector index, axis of length 1]
        jump_states = tf.squeeze(jump_states, axis=-1)
        states = tf.squeeze(states, axis=-1)

        # first calculate the no-jump trajectory, we do this separately once so that we only sample trajectories with jumps below
        # see https://link.aps.org/doi/10.1103/PhysRevA.99.052327 for more details
        for k in tf.range(blocks.shape[0]):
            states = tf.einsum(
                "mij,msj->msi", blocks[k, ...], states
            )  # t: trajectories, m: multistart, s:multiple states

        p_no_jump = tf.cast(tf.einsum("msi,msi->ms", tf.math.conj(states), states), dtype=tf.float32)
        no_jump_overlaps = tf.einsum("si...,msi->ms...", self.target_states_conj, states) # average over inital states here to implement "coherent" state transfer
        mean_overlaps = tf.reduce_mean(no_jump_overlaps, axis=[1]) # average over initial states to implement "coherent" state transfer
        no_jump_fidelities = tf.math.conj(mean_overlaps) * mean_overlaps
        no_jump_fidelities = tf.cast(tf.squeeze(no_jump_fidelities), dtype=tf.float32) # note that this fidelity is F_no_jump * p_no_jump

        # now calculate the trajectories with at least 1 jump

        R = tf.random.uniform(jump_states.shape[0:3], minval=tf.stack([p_no_jump] * jump_states.shape[0]), maxval=1.0, dtype=tf.float32)

        for k in tf.range(blocks.shape[0]):
            jump_states = tf.einsum(
                "mij,tmsj->tmsi", blocks[k, ...], jump_states
            )  # t: trajectories, m: multistart, s:multiple states
        
            # calculate norms to compare to random numbers R and decide on jumps
            # we'll do this by making a mask of the trajectories that will "jump"
            norms = tf.einsum("tmsi,tmsi->tms", tf.math.conj(jump_states), jump_states)

            mask = (R > tf.cast(norms, dtype=tf.float32)) # one indicates to insert a jump in that trajectory, 0 does nothing
            mask = tf.cast(mask, dtype=tf.complex64) # need to cast the mask to type complex64

            jump_op = tf.einsum("tms,ij->tmsij", mask, self.jump_ops) + tf.einsum("tms,ij->tmsij", 1 - mask, tf.eye(self.N, dtype=tf.complex64)) # this combines jump_op and identity to produce a masked jump operator on all trajs
            jump_states = tf.einsum("tmsij,tmsj->tmsi", jump_op, jump_states) # only support one jump op at the moment

            # now selectively re-normalize the states where we applied a jump operator
            norms = tf.math.sqrt(tf.einsum("tmsi,tmsi->tms", tf.math.conj(jump_states), jump_states))
            masked_norms = mask * norms + (1 - mask) * tf.ones_like(mask) # mask so that we only renormalize states where we just inserted a jump

            jump_states = tf.einsum("tmsi,tms->tmsi", jump_states, 1 / masked_norms)
            
            R_new = tf.random.uniform(jump_states.shape[0:3], minval=0, maxval=1.0, dtype=tf.float32) # generate new random numbers
            R = tf.cast(mask, dtype=tf.float32) * R_new + (1 - tf.cast(mask, dtype=tf.float32)) * R # selectively update the random numbers based on the mask
        
        # renormalize all states now that we're done
        norms = tf.einsum("tmsi,tmsi->tms", tf.math.conj(jump_states), jump_states)
        jump_states = tf.einsum("tmsi,tms->tmsi", jump_states, 1 / tf.math.sqrt(norms))
        
        # average over inital states here to implement "coherent" state transfer. also weight by probability of 1-p_no_jump
        overlaps = tf.einsum("si...,tmsi,ms->tms...", self.target_states_conj, jump_states, tf.sqrt(1 - tf.cast(p_no_jump, dtype=tf.complex64)))
        mean_overlaps = tf.reduce_mean(overlaps, axis=[2]) # average over initial states to implement "coherent" state transfer
        jump_fidelities = tf.math.conj(mean_overlaps) * mean_overlaps
        jump_fidelities = tf.reduce_mean(jump_fidelities, axis=[0]) # average over trajectories.
        jump_fidelities = tf.cast(tf.squeeze(jump_fidelities), dtype=tf.float32)
        joint_fidelities = no_jump_fidelities + jump_fidelities

        if self.use_conditional_fid:
            max_joint_fid = tf.math.reduce_max(joint_fidelities)
            success_prob_no_jump = tf.cast(tf.einsum('msi,ij,msj->ms', tf.math.conj(states), self.success_op, states), dtype=tf.float32)
            success_prob_jumps = tf.cast(tf.einsum('tmsi,ij,tmsj,ms->tms', 
                                                   tf.math.conj(jump_states), self.success_op, jump_states, 1 - tf.cast(p_no_jump, dtype=tf.complex64)), 
                                         dtype=tf.float32)
            success_prob_jumps = tf.reduce_mean(success_prob_jumps, axis=[0])
            success_probs = tf.reduce_mean(success_prob_no_jump + success_prob_jumps, axis=[1])
            return joint_fidelities / (1 - (1 - success_probs) * tf.keras.activations.relu((max_joint_fid - self.threshold_start) / (self.threshold_end - self.threshold_start), max_value=1))

        return joint_fidelities