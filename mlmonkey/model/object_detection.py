import sys
import os

from mlmonkey.task import Task
from mlmonkey.utils import datasets

tools_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'object_detection', 'pytorch')
config_defaule_file = os.path.join(tools_path, 'configs', 'e2e_mask_rcnn_R_50_FPN_1x.yaml')


class ObjectDetectionModel(Task):
    def __init__(self, **kwargs):
        super(Task, self).__init__(**kwargs)

        self.config_file = kwargs.pop('config_file', config_defaule_file)
        self.train_batch = kwargs.pop('train_batch', 2)
        self.test_batch = kwargs.pop('test_batch', 1)
        self.bast_lr = kwargs.pop('base_lr', 0.0025)
        self.max_iter = kwargs.pop('max_iter', 720000)
        self.lr_decay_steps = kwargs.pop('lr_decay_steps', '(480000, 640000)')

    def task_arguments(self):
        train_exec = 'train_mlperf.py'
        args = [sys.executable,
                os.path.join(tools_path, 'tools', train_exec),
                ]

        if self.config_file is not None:
            args.append('--config-file=%s' % self.config_file)

        if self.train_batch is not None:
            args.append('SOLVER.IMS_PER_BATCH=%s' % self.train_batch)

        if self.test_batch is not None:
            args.append('TEST.IMS_PER_BATCH=%s' % self.test_batch)

        if self.bast_lr is not None:
            args.append('SOLVER.BASE_LR' % self.bast_lr)

        if self.max_iter is not None:
            args.append('SOLVER.MAX_ITER=%s' % self.max_iter)

        if self.lr_decay_steps is not None:
            args.append('SOLVER.STEPS=%s' % self.lr_decay_steps)

        return args

    def download_dataset(self):
        data_folder = os.path.join(tools_path, 'pytorch', 'datasets', 'coco')
        os.makedirs(data_folder)

        coco_data = ['https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz',
                     'http://images.cocodataset.org/zips/train2014.zip',
                     'http://images.cocodataset.org/zips/val2014.zip',
                     'http://images.cocodataset.org/annotations/annotations_trainval2014.zip']
        for url in coco_data:
            file = datasets.download(url, model='object_detection', assign_path=data_folder)
            datasets.extract(file)

    def get_dataset(self):
        datasets_2014 = ['datasets/coco/annotations', 'datasets/coco/train2014', 'datasets/coco/val2014']
        datasets_2017 = ['datasets/coco/annotations', 'datasets/coco/train2017', 'datasets/coco/val2017']

        for path in map(lambda p: os.path.join(tools_path, p), datasets_2014.copy()):
            if os.path.isdir(path) is False:
                datasets_2014 = False
                break

        for path in map(lambda p: os.path.join(tools_path, p), datasets_2017.copy()):
            if os.path.isdir(path) is False:
                datasets_2017 = False
                break

        if datasets_2014 is False or datasets_2017 is False:
            return False

        return True

    def verify_dataset(self):
        pass
