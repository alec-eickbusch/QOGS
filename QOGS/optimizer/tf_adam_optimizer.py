# note: timestamp can't use "/" character for h5 saving.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf
from contextlib import ExitStack

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # supress warnings
import h5py

print(
    "\nNeed tf version 2.3.0 or later. Using tensorflow version: "
    + tf.__version__
    + "\n"
)
import QOGS.optimizer.tf_quantum as tfq
from QOGS.optimizer.visualization import VisualizationMixin
from QOGS.optimizer.GateSynthesizer import GateSynthesizer
import datetime
import time


class AdamOptimizer(VisualizationMixin):
    """
    Here we optimize the loss function using the Adam optimizer with a gradient 
    generated by autodifferentiation.
    """

    def __init__(self, gatesynth: GateSynthesizer):
        """
        Here, we create references to all the objects we need in the ``GateSynthesizer`` object
        which defines the optimization problem.

        Parameters
        -----------
        gatesynth  :   an instance of the ``GateSynthesizer`` class.

        """
        self.timestamps = gatesynth.timestamps
        self.parameters = gatesynth.parameters
        self.opt_vars = gatesynth.opt_vars
        self.loss_fun = gatesynth.loss_fun
        self.callback_fun = gatesynth.callback_fun
        self.batch_fidelities = gatesynth.batch_fidelities
        self.mask = gatesynth.optimization_mask
        self._save_termination_reason = gatesynth._save_termination_reason
        self.print_info = gatesynth.print_info
        self.filename = gatesynth.filename

        return

    @tf.function
    def entry_stop_gradients(self, vars, mask_var):
        """
        This function masks certain trainable parameters from the gradient calculator.
        This is useful if one of the block parameters is a constant.

        Parameters
        -----------
        vars    :   List of tf.variable. This list should be of the same length as
                    self.opt_vars.
        mask    :   List of masks of the same length as target.

        Returns
        -----------
        list of tf.tensor with some block parameters masked out of the gradient calculation
        """

        mask_list = {}
        for key, value in vars.items():
            mask_h = tf.abs(mask_var[key] - 1)
            mask_list[key] = tf.stop_gradient(mask_h * value) + mask_var[key] * value
        return mask_list

    def optimize(self, logdir=None):

        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        self.timestamps.append(timestamp)
        print("Start time: " + timestamp)
        # start time
        start_time = time.time()
        optimizer = tf.optimizers.Adam(self.parameters["learning_rate"])
        if logdir is not None:
            tf.profiler.experimental.start(logdir)

        initial_fids = self.batch_fidelities(self.opt_vars)
        fids = initial_fids
        self.callback_fun(fids, 0, 0, timestamp, start_time)
        try:  # will catch keyboard inturrupt
            for epoch in range(self.parameters["epochs"] + 1)[1:]:
                for substep in range(self.parameters["epoch_size"]):
                    with ExitStack() as stack:
                        if logdir is not None:
                            stack.enter_context(
                                tf.profiler.experimental.Trace(
                                    "opt_trace",
                                    step_num=substep
                                    + epoch * self.parameters["epoch_size"],
                                    _r=1,
                                )
                            )
                        tape = stack.enter_context(tf.GradientTape())
                        masked_vars = self.entry_stop_gradients(
                            self.opt_vars, self.mask
                        )
                        new_fids = self.batch_fidelities(masked_vars)
                        new_loss = self.loss_fun(new_fids)
                        dloss_dvar = tape.gradient(
                            new_loss, list(self.opt_vars.values())
                        )
                    optimizer.apply_gradients(
                        zip(dloss_dvar, list(self.opt_vars.values()))
                    )  # note that optimizer is not included in the profiler; it's not resource-intensive
                dfids = new_fids - fids
                fids = new_fids
                self.callback_fun(fids, dfids, epoch, timestamp, start_time)
                condition_fid = tf.greater(fids, self.parameters["term_fid"])
                condition_dfid = tf.greater(dfids, self.parameters["dfid_stop"])
                if tf.reduce_any(condition_fid):
                    print("\n\n Optimization stopped. Term fidelity reached.\n")
                    termination_reason = "term_fid"
                    break
                if not tf.reduce_any(condition_dfid):
                    print("\n max dFid: %6f" % tf.reduce_max(dfids).numpy())
                    print("dFid stop: %6f" % self.parameters["dfid_stop"])
                    print(
                        "\n\n Optimization stopped.  No dfid is greater than dfid_stop\n"
                    )
                    termination_reason = "dfid"
                    break
        except KeyboardInterrupt:
            print("\n max dFid: %6f" % tf.reduce_max(dfids).numpy())
            print("dFid stop: %6f" % self.parameters["dfid_stop"])
            print("\n\n Optimization stopped on keyboard interrupt")
            termination_reason = "keyboard_interrupt"

        if epoch == self.parameters["epochs"]:
            termination_reason = "epochs"
            print(
                "\n\nOptimization stopped.  Reached maximum number of epochs. Terminal fidelity not reached.\n"
            )
        self._save_termination_reason(timestamp, termination_reason)
        timestamp_end = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        elapsed_time_s = time.time() - start_time
        epoch_time_s = elapsed_time_s / epoch
        step_time_s = epoch_time_s / self.parameters["epochs"]
        self.print_info()
        print("all data saved as: " + self.filename)
        print("termination reason: " + termination_reason)
        print("optimization timestamp (start time): " + timestamp)
        print("timestamp (end time): " + timestamp_end)
        print("elapsed time: " + str(datetime.timedelta(seconds=elapsed_time_s)))
        print(
            "Time per epoch (epoch size = %d): " % self.parameters["epoch_size"]
            + str(datetime.timedelta(seconds=epoch_time_s))
        )
        print(
            "Time per Adam step (N_multistart = %d): "
            % (self.parameters["N_multistart"])
            + str(datetime.timedelta(seconds=step_time_s))
        )
        print(END_OPT_STRING)
        if logdir is not None:
            tf.profiler.experimental.stop()
        return timestamp

    # if append is True, it will assume the dataset is already created and append only the
    # last aquired values to it.
    # TODO: if needed, could use compression when saving data.
