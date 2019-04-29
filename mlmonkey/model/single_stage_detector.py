import sys
import os
import subprocess

from mlmonkey.task import Task

tools_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'single_statge_detector')


class SingleStageDetectorModel(Task):
    def __init__(self, **kwargs):
        super(Task, self).__init__(**kwargs)

        self.epochs = kwargs.pop('epochs', 70)
        self.warmup_factor = kwargs.pop('warmup_factor', 0)
        self.lr = kwargs.pop('learning_rate', 2.5e-3)
        self.no_save = kwargs.pop('no_save', True)
        self.threshold = kwargs.pop('threshold', 0.23)

    def task_arguments(self):
        train_exec = 'train.py'
        args = [sys.executable,
                os.path.join(tools_path, 'tools', train_exec),
                ]

        if self.epochs is not None:
            args.append('--epochs=%s' % self.epochs)

        if self.warmup_factor is not None:
            args.append('--warmup_factor=%s' % self.warmup_factor)

        if self.lr is not None:
            args.append('--lr=%s' % self.lr)

        if self.no_save is not None:
            args.append('--no_save')

        if self.threshold is not None:
            args.append('--threshold=%s' % self.threshold)

        return args

    def download_dataset(self):
        download_exec = 'librispeech.py'
        args = [sys.executable,
                os.path.join(tools_path, 'data', download_exec),
                ]

        ret = subprocess.check_output(args)

    def get_dataset(self):
        datasets_2017 = ['datasets/coco/annotations', 'datasets/coco/train2017', 'datasets/coco/val2017']

        for path in map(lambda p: os.path.join(tools_path, p), datasets_2017.copy()):
            if os.path.isdir(path) is False:
                datasets_2017 = False
                break

        if datasets_2017 is False:
            return False

        return True

    def verify_dataset(self):
        pass
