import gevent
import gevent.event
import uuid
import time
import traceback
import os
import shutil
import signal
import pickle
import platform
import subprocess

from mlmonkey.status import Status
from mlmonkey import constants


class Task:
    def __init__(self, job_dir, name):
        self.job_dir = job_dir
        self.job_id = os.path.basename(job_dir)
        self._id = str(uuid.uuid4())
        self.name = name
        self.status = Status.INIT
        self.aborted = gevent.event.Event()
        self.SAVE_FILE = 'status.pickle'

    def name(self):
        """
        Return task's name.
        :return: name
        """
        return self.name

    def id(self):
        """
        Return task's id.
        :return: id
        """
        return self._id

    def dir(self):
        """
        Return task's dir
        :return:
        """
        return os.path.join(self.job_dir, self.id())

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

    def abort(self):
        """
        Abort the Task
        """
        if self.status.is_running():
            self.aborted.set()
            self.status = Status.ABORT

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
        """
        Returns args used by subprocess.Popen to execute the task
        Returns False if the args cannot be set properly
        """
        raise NotImplementedError

    def json_dict(self):
        return {
            'id': self.id(),
            'status': self.status.name
        }

    def download_dataset(self):
        """
        download dataset
        Raises exceptions
        """
        raise NotImplementedError

    def verify_dataset(self):
        """
        verity dataset
        Raises exceptions
        """
        raise NotImplementedError

    def get_dataset(self):
        """
        get the dataset
        Raises exceptions
        """
        raise NotImplementedError

    def before_run(self):
        """
        Called before run() executes
        """
        if self.get_dataset():
            return True
        return False

    def run(self, resources):
        """
        Execute the task
        Arguments:
        resources -- the resources assigned by the scheduler for this task
        """
        check_exec = self.before_run()

        if check_exec is False:
            self.status = Status.ERROR
            return False

        env = os.environ.copy()
        args = self.task_arguments(resources, env)
        if not args:
            self.logger.error('Could not create the arguments for Popen')
            self.status = Status.ERROR
            return False
        # Convert them all to strings
        args = [str(x) for x in args]

        self.logger.info('%s task started.' % self.name())
        self.status = Status.RUN

        unrecognized_output = []

        import sys
        env['PYTHONPATH'] = os.pathsep.join(['.', self.job_dir, env.get('PYTHONPATH', '')] + sys.path)

        # https://docs.python.org/2/library/subprocess.html#converting-argument-sequence
        if platform.system() == 'Windows':
            args = ' '.join(args)
            self.logger.info('Task subprocess args: "{}"'.format(args))
        else:
            self.logger.info('Task subprocess args: "%s"' % ' '.join(args))

        self.p = subprocess.Popen(args,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  cwd=self.job_dir,
                                  close_fds=False if platform.system() == 'Windows' else True,
                                  env=env,
                                  )

        try:
            sigterm_time = None  # When was the SIGTERM signal sent
            sigterm_timeout = 120  # When should the SIGKILL signal be sent
            while self.p.poll() is None:
                for line in utils.nonblocking_readlines(self.p.stdout):
                    if self.aborted.is_set():
                        if sigterm_time is None:
                            # Attempt graceful shutdown
                            self.p.send_signal(signal.SIGTERM)
                            sigterm_time = time.time()
                            self.status = Status.ABORT
                        break

                    if line is not None:
                        # Remove whitespace
                        line = line.strip()

                    if line:
                        if not self.process_output(line):
                            # self.logger.warning('%s unrecognized output: %s' % (self.name(), line.strip()))
                            unrecognized_output.append(line)
                    else:
                        time.sleep(0.05)
                if sigterm_time is not None and (time.time() - sigterm_time > sigterm_timeout):
                    self.p.send_signal(signal.SIGKILL)
                    self.logger.warning('Sent SIGKILL to task "%s"' % self.name())
                time.sleep(0.01)
        except:
            self.p.terminate()
            self.after_run()
            raise

        self.after_run()

        if self.status != Status.RUN:
            return False
        elif self.p.returncode != 0:
            self.errorcode = self.p.returncode
            self.logger.error('%s task failed with error code %d' % (self.name(), self.errorcode))
            if self.exception is None:
                # _, error = self.p.communicate()
                self.exception = 'error code %d' % self.p.returncode
                if unrecognized_output:
                    last_line = unrecognized_output[-1]
                    if self.traceback is None:
                        self.traceback = '\n'.join(unrecognized_output)
                    else:
                        self.traceback = self.traceback + ('\n'.join(unrecognized_output))
                    self.exception = last_line
            self.logger.error(self.exception)

            self.after_runtime_error()
            self.status = Status.ERROR
            return False
        else:
            self.logger.info('%s task completed.' % self.name())
            self.status = Status.DONE
            # if unrecognized_output:
            #    self.logger.warning('unrecognized output: %s' % ('\n'.join(unrecognized_output)))
            return True

    def process_output(self, line):
        """
        Process a line of output from the task
        Returns True if the output was able to be processed
        Arguments:
        line -- a line of output
        """
        raise NotImplementedError

    def after_run(self):
        """
        Called after run() executes
        """
        raise NotImplementedError
