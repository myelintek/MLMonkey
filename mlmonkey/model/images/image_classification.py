import sys
import os
import random

from mlmonkey.task import Task

tools_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'image_classification', 'tensorflow',
                          'official', 'resnet')


class ImageClassificationModel(Task):
    def __init__(self, **kwargs):
        super(Task, self).__init__(**kwargs)
        self.random_seed = kwargs.pop('random_seed', random.randint(1, 5))
        self.data_dir = kwargs.pop('data_dir ', None)
        self.num_gpus = kwargs.pop('num_gpus ', 1)
        self.batch_size = kwargs.pop('batch_size', 64)
        self.model_dir = kwargs.pop('model', '/tmp/resnet_imagenet_%s' % self.random_seed)
        self.train_epochs = kwargs.pop('train_epochs', 1)
        self.stop_threshold = kwargs.pop('stop_threshold', 0.79)
        self.version = kwargs.pop('version', 1)
        self.resnet_size = kwargs.pop('resnet_size', 50)
        self.epochs_between_evals = kwargs.pop('epochs_between_evals', 4)

    def task_arguments(self):
        train_exec = 'imagenet_main.py'
        args = [sys.executable, tools_path, train_exec]

        if self.random_seed is not None:
            args.append('--random_seed=%s' % self.random_seed)

        if self.data_dir is not None:
            args.append('--data_dir=%s' % self.data_dir)
        else:
            # If not specified, synthetic data will be used.
            args.append('--data_dir=None')

        if self.num_gpus is not None:
            args.append('--num_gpus=%s' % self.num_gpus)

        if self.batch_size is not None:
            args.append('--batch_size=%s' % self.batch_size)

        if self.model_dir is not None:
            args.append('--model_dir=%s' % self.model_dir)

        if self.train_epochs is not None:
            args.append('--train_epochs=%s' % self.train_epochs)

        if self.epochs_between_evals is not None:
            args.append('--epochs_between_evals=%s' % self.epochs_between_evals)

        if self.stop_threshold is not None:
            args.append('--stop_threshold=%s' % self.stop_threshold)

        if self.resnet_size is not None:
            args.append('--resnet_size=%s' % self.resnet_size)

        if self.version is not None:
            args.append('--version=%s' % self.version)

        return args

    def download_dataset(self):
        pass

    def verify_dataset(self):
        pass

    def get_dataset(self):
        return True
