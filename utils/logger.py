from time import time

from os.path import join, dirname

_log_path = join(dirname(__file__), '../logs')

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class TFLogger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.compat.v1.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


# high level wrapper for tf_logger.TFLogger
class Logger():
    def __init__(self, args, update_frequency=10):
        self.current_epoch = 0
        self.max_epochs = args.epochs
        self.last_update = time()
        self.start_time = time()
        self._clean_epoch_stats()
        self.update_f = update_frequency
        folder, logname = self.get_name_from_args(args)
        log_path = join(_log_path, folder, logname)
        if args.tf_logger:
            self.tf_logger = TFLogger(log_path)
            print("Saving to %s" % log_path)
        else:
            self.tf_logger = None
        self.current_iter = 0

    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.last_update = time()
        self.lrs = learning_rates
        print("New epoch - lr: %s" % ", ".join([str(lr) for lr in self.lrs]))
        self._clean_epoch_stats()
        if self.tf_logger:
            for n, v in enumerate(self.lrs):
                self.tf_logger.scalar_summary("aux/lr%d" % n, v, self.current_iter)

    def log(self, it, iters, losses, samples_right, total_samples):
        self.current_iter += 1
        loss_string = ", ".join(["%s : %.3f" % (k, v) for k, v in losses.items()])
        for k, v in samples_right.items():
            past = self.epoch_stats.get(k, 0.0)
            self.epoch_stats[k] = past + v
        self.total += total_samples
        acc_string = ", ".join(["%s : %.2f" % (k, 100 * (v / total_samples)) for k, v in samples_right.items()])
        if it % self.update_f == 0:
            print("%d/%d of epoch %d/%d %s - acc %s [bs:%d]" % (it, iters, self.current_epoch, self.max_epochs, loss_string,
                                                                acc_string, total_samples))
            # update tf log
            if self.tf_logger:
                for k, v in losses.items(): self.tf_logger.scalar_summary("train/loss_%s" % k, v, self.current_iter)

    def _clean_epoch_stats(self):
        self.epoch_stats = {}
        self.total = 0

    def log_test(self, phase, accuracies):
        print("Accuracies on %s: " % phase + ", ".join(["%s : %.2f" % (k, v * 100) for k, v in accuracies.items()]))
        if self.tf_logger:
            for k, v in accuracies.items(): self.tf_logger.scalar_summary("%s/acc_%s" % (phase, k), v, self.current_iter)

    def save_best(self, val_test, best_test):
        print("It took %g" % (time() - self.start_time))
        if self.tf_logger:
            for x in range(10):
                self.tf_logger.scalar_summary("best/from_val_test", val_test, x)
                self.tf_logger.scalar_summary("best/max_test", best_test, x)

    @staticmethod
    def get_name_from_args(args):
        folder_name = "%s_to_%s" % ("-".join(sorted(args.source)), args.target)
        if args.folder_name:
            folder_name = join(args.folder_name, folder_name)
        name = "eps%d_bs%d_lr%g_class%d" % (args.epochs, args.batch_size, args.learning_rate, args.n_classes)
        # if args.ooo_weight > 0:
        #     name += "_oooW%g" % args.ooo_weight
        if args.train_all:
            name += "_TAll"
        if args.bias_whole_image:
            name += "_bias%g" % args.bias_whole_image
        if args.classify_only_sane:
            name += "_classifyOnlySane"
        if args.TTA:
            name += "_TTA"
        try:
            name += "_entropy%g_jig_tW%g" % (args.entropy_weight, args.target_weight)
        except AttributeError:
            pass
        if args.suffix:
            name += "_%s" % args.suffix
        name += "_%d" % int(time() % 1000)
        return folder_name, name
