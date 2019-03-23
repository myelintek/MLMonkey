from status import Status
import gevent.event
import sys
import os


class Scenario(object):

    def __init__(self,
                 id,
                 name,
                 data_dir=None,
                 model=None,
                 num_gpus=1,
                 batch_size=32,
                 variable_update='replicated',
                 fp16=True,
                 optimizer='sgd',
                 data_format='NCHW',
                 num_epochs=None):
        self._id = id
        self._name = name
        self.data_dir = data_dir
        self.model = model
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.variable_update = variable_update
        self.fp16 = fp16
        self.optimizer = optimizer
        self.data_format = data_format
        self.num_epochs = num_epochs

    def id(self):
        return self._id

    def name(self):
        return self._name

    def path(self):
        if not self.id():
            return None

        path = os.path.join(os.environ('dir'), self.id())

        return str(path).replace("\\", "/")

    def json_dict(self):
        d = {
            'id': self.id(),
            'name': self.name(),
            'data_dir': self.data_dir,
            'num_gpus': self.num_gpus,
            'batch_size': self.batch_size,
            'model': self.model,
            'variable_update': self.variable_update,
            'fp16': self.fp16,
            'optimizer': self.optimizer,
            'data_format': self.data_format,
            'num_epochs': self.num_epochs
        }
        return d


class Task(Scenario):
    def __init__(self,
                 id,
                 name,
                 data_dir=None,
                 model=None,
                 num_gpus=1,
                 batch_size=32,
                 variable_update='replicated',
                 fp16=True,
                 optimizer='sgd',
                 data_format='NCHW',
                 num_epochs=None,
                 status=Status.INIT):
        super(Task, self).__init__(id=id, name=name, data_dir=data_dir, model=model, num_gpus=num_gpus,
                                   batch_size=batch_size, variable_update=variable_update, fp16=fp16,
                                   optimizer=optimizer, data_format=data_format, num_epochs=num_epochs)
        self.status = status
        self.aborted = gevent.event.Event()

    def task_arguments(self):
        param = self.json_dict()
        train_exec = 'tf_cnn_benchmarks.py'
        args = [sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'tf_cnn_benchmarks', train_exec),
                ]

        if param.data_dir is not None:
            args.append('--data_dir=%s' % param.data_dir)
        else:
            # If not specified, synthetic data will be used.
            args.append('--data_dir=None')

        if param.num_gpus is not None:
            args.append('--num_gpus=%s' % param.num_gpus)

        if param.batch_size is not None:
            args.append('--batch_size=%s' % param.batch_size)

        if param.model is not None:
            args.append('--model=%s' % param.model)

        if param.variable_update is not None:
            args.append('--variable_update=%s' % param.variable_update)

        if param.fp16 is not None:
            args.append('--fp16=%s' % param.fp16)

        if param.optimizer is not None:
            args.append('--optimizer=%s' % param.optimizer)

        if param.data_format is not None:
            args.append('--data_format=%s' % param.data_format)

        if param.num_epochs is not None:
            args.append('--num_epochs=%s' % param.num_epochs)

        return args

    def before_run(self):

        if not self.status.is_ready():
            return False

        train_task = self.task_arguments()
        if not train_task:
            self.status = Status.ERROR
            return False

        self.status = Status.WAIT
        return train_task

    def run(self):
        train_task = self.before_run()

        self.p = gevent.spawn(train_task)
        self.status = Status.RUN

    def after_run(self):
        if self.status is not Status.ERROR:
            pass

    def abort(self):
        if self.status.is_running():
            self.status = Status.ABORT
            self.aborted.set()

        return True
