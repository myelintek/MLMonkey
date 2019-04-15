import gevent.event
import sys
import os
import uuid
import pickle
import shutil
import traceback
from mlmonkey import constants
from mlmonkey.status import Status


class Scenario:
    def __init__(self,
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
        self._id = uuid.uuid4()
        self.name = name
        self.data_dir = data_dir
        self.model = model
        self.num_gpus = num_gpus
        self.batch_size = batch_size
        self.variable_update = variable_update
        self.fp16 = fp16
        self.optimizer = optimizer
        self.data_format = data_format
        self.num_epochs = num_epochs

        self.status = status
        self.aborted = gevent.event.Event()

        self._dir = os.path.join(constants.JOBS_DIR, self._id)
        self.SAVE_FILE = 'status.pickle'

        os.mkdir(self._dir)

    def id(self):
        return self._id

    def dir(self):
        """getter for _dir"""
        return self._dir

    def path(self, filename, relative=False):
        """
        Returns a path to the given file
        Arguments:
        filename -- the requested file
        Keyword arguments:
        relative -- If False, return an absolute path to the file
                    If True, return a path relative to the jobs directory
        """
        if not filename:
            return None
        if os.path.isabs(filename):
            path = filename
        else:
            path = os.path.join(self._dir, filename)
            if relative:
                path = os.path.relpath(path, constants.JOBS_DIR)
        return str(path).replace("\\", "/")

    def json_dict(self):
        d = {
            'id': self.id(),
            'name': self.name,
            'data_dir': self.data_dir,
            'num_gpus': self.num_gpus,
            'batch_size': self.batch_size,
            'model': self.model,
            'variable_update': self.variable_update,
            'fp16': self.fp16,
            'optimizer': self.optimizer,
            'data_format': self.data_format,
            'num_epochs': self.num_epochs,
            'status': self.status
        }
        return d

    def load(self):
        pass

    def save(self):
        """
        Saves the job to disk as a pickle file
        Suppresses errors, but returns False if something goes wrong
        """
        try:
            # use tmpfile so we don't abort during pickle dump (leading to EOFErrors)
            tmpfile_path = self.path(self.SAVE_FILE + '.tmp')
            with open(tmpfile_path, 'wb') as tmpfile:
                pickle.dump(self, tmpfile)
            file_path = self.path(self.SAVE_FILE)
            shutil.move(tmpfile_path, file_path)
            return True
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print('Caught %s while saving job %s: %s' % (type(e).__name__, self.id(), e))
            traceback.print_exc()
        return False

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


