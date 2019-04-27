import sys
import os

from mlmonkey.task import Task


class TfBenchmarkModel(Task):
    def __init__(self, **kwargs):
        super(Task, self).__init__(**kwargs)
        self.data_dir = kwargs.pop('data_dir', None)
        self.num_gpus = kwargs.pop('num_gpus', None)
        self.batch_size = kwargs.pop('batch_size', None)
        self.model = kwargs.pop('model', None)
        self.variable_update = kwargs.pop('variable_update', None)
        self.fp16 = kwargs.pop('fp16', None)
        self.optimizer = kwargs.pop('optimizer', None)
        self.data_format = kwargs.pop('data_format', None)
        self.num_epochs = kwargs.pop('num_epochs', None)

    def task_arguments(self):
        train_exec = 'tf_cnn_benchmarks.py'
        args = [sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'tf_cnn_benchmarks', train_exec),
                ]

        if self.data_dir is not None:
            args.append('--data_dir=%s' % self.data_dir)
        else:
            # If not specified, synthetic data will be used.
            args.append('--data_dir=None')

        if self.num_gpus is not None:
            args.append('--num_gpus=%s' % self.num_gpus)

        if self.batch_size is not None:
            args.append('--batch_size=%s' % self.batch_size)

        if self.model is not None:
            args.append('--model=%s' % self.model)

        if self.variable_update is not None:
            args.append('--variable_update=%s' % self.variable_update)

        if self.fp16 is not None:
            args.append('--fp16=%s' % self.fp16)

        if self.optimizer is not None:
            args.append('--optimizer=%s' % self.optimizer)

        if self.data_format is not None:
            args.append('--data_format=%s' % self.data_format)

        if self.num_epochs is not None:
            args.append('--num_epochs=%s' % self.num_epochs)

        return args

    def download_dataset(self):
        pass

    def verify_dataset(self):
        pass

    def get_dataset(self):
        return True
