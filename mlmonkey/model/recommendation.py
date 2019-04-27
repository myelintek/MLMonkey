import sys
import os
import subprocess

from mlmonkey.task import Task
from mlmonkey.utils import datasets
from mlmonkey.log import logger

tools_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'recommendation', 'pytorch')


class ObjectDetectionModel(Task):
    def __init__(self, **kwargs):
        super(Task, self).__init__(**kwargs)

        self.lr = kwargs.pop('learning_rate', 0.0002)
        self.batch_size = kwargs.pop('batch_size', 65536)
        self.layers = kwargs.pop('layers', [256, 256, 128, 64])
        self.factors = kwargs.pop('factors', 64)
        self.seed = kwargs.pop('seed', 1)
        self.threshold = kwargs.pop('threshold', 1.0)
        self.user_scaling = kwargs.pop('user_scaling', 16)
        self.item_scaling = kwargs.pop('item_scaling', 32)
        self.cpu_dataloader = kwargs.pop('cpu_dataloader', True)
        self.random_negatives = kwargs.pop('random_negatives', True)

    def task_arguments(self):
        train_exec = 'ncf.py'
        args = [sys.executable,
                os.path.join(tools_path, 'pytorch', train_exec),
                ]

        if self.lr is not None:
            args.append('--l=%s' % self.lr)

        if self.batch_size is not None:
            args.append('--b=%s' % self.batch_size)

        if self.layers is not None:
            args.append('--layers=%s' % self.layers)

        if self.factors is not None:
            args.append('--f=%s' % self.factors)

        if self.seed is not None:
            args.append('--seed=%s' % self.seed)

        if self.threshold is not None:
            args.append('--threshold=%s' % self.threshold)

        if self.user_scaling is not None:
            args.append('--user_scaling=%s' % self.user_scaling)

        if self.item_scaling is not None:
            args.append('--item_scaling=%s' % self.item_scaling)

        if self.cpu_dataloader is not None:
            args.append('--cpu_dataloader')

        if self.random_negatives is not None:
            args.append('--random_negatives')

        return args

    def download_dataset(self):
        download_url = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
        file = datasets.download(download_url, model='recommendation')
        datasets.extract(file)

    def verify_dataset(self):
        hash_code = '<(echo "MD5 (ml-20m.zip) = cd245b17a1ae2cc31bb14903e1204af3")'
        cmd = 'diff < (md5sum ml-20m.zip) %s > /dev/null' % hash_code
        check = subprocess.check_output(cmd)
        if check:
            logger.debug('verify recommendation dataset PASS!')
        else:
            logger.warning('verify recommendation dataset FAIL!')

    def get_dataset(self):

        return True
