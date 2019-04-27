import sys
import os
import random
import hashlib

from mlmonkey.task import Task
from mlmonkey.utils import datasets
from mlmonkey.log import logger

tools_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'sentiment_analysis')


class SentimentAnalysisModel(Task):
    def __init__(self, **kwargs):
        super(Task, self).__init__(**kwargs)

        self.seed = kwargs.pop('seed', random.randint(0, 10))
        self.model = kwargs.pop('model', 'conv')
        self.target_quality = kwargs.pop('quality', 90.6)

    def task_arguments(self):
        train_exec = 'train.py'
        args = [sys.executable,
                os.path.join(tools_path, 'paddle', train_exec),
                ]

        if self.seed is not None:
            args.append('--seed=%s' % self.seed)

        if self.model is not None:
            args.append('--model=%s' % self.model)

        if self.target_quality is not None:
            args.append('--target_quality=%s' % self.target_quality)

        return args

    def download_dataset(self):
        url = 'http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz'
        data_folder = os.path.expanduser('~/.cache/paddle/dataset/imdb')

        file = datasets.download(url=url, model='sentiment_analysis', assign_path=data_folder)
        datasets.extract(file)

    def verify_dataset(self):
        MD5 = '7c2ac02c03563afcf9b574c7e56c153a'
        url = 'http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz'
        data_folder = os.path.expanduser('~/.cache/paddle/dataset/imdb')
        path = os.path.join(url, data_folder.split('/')[-1])
        CHUNK_SIZE = 4096
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                hash_md5.update(chunk)

        if hash_md5.hexdigest() == MD5:
            logger.debug('verify sentiment_analysis dataset PASS!')
        else:
            logger.warning('verify sentiment_analysis dataset FAIL!')

    def get_dataset(self):
        return True
