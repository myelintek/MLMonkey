from collections import OrderedDict

import gevent
import os
import json
import time

from mlmonkey import scenario
from mlmonkey import constants
from mlmonkey.status import Status


class Scheduler:
    def __init__(self):
        self._tasks = OrderedDict()

        self.shutdown = gevent.event.Event()

    def load_past_tasks(self):
        file = constants.SCENARIOS_JSON
        with open(file, 'r') as json_file:
            datas = json.load(json_file)
        for d in datas:
            task = scenario.Scenario(
                name=d['name'],
                data_dir=d['data_dir'],
                num_gpus=d['num_gpus'],
                batch_size=d['batch_size'],
                model=d['model'],
                variable_update=d['variable_update'],
                fp16=d['fp16'],
                optimizer=d['optimizer'],
                data_format=d['data_format'],
                num_epochs=d['num_epochs'])
            self._tasks[task.id()] = task

    def add_task(self, task):
        self._tasks[task.id()] = task
        self.json_handler(task, action='w')

    def delete_task(self, id):
        task = self.get_task(id)
        self.json_handler(task, action='d')

    def json_handler(self, task, action='r'):

        file = constants.SCENARIOS_JSON
        if action is 'w':
            d = {
                'id': task.id(),
                'name': task.name,
                'status': task.status,
                'scenario': task.scenario.json_dict()
            }
            if os.path.isfile(file):
                with open(file, 'r') as json_file:
                    data = json.load(json_file)
                    data.append(d)
                with open(file, 'w+') as json_file:
                    json.dump(data, json_file, indent=4)
            else:
                data = []
                data.append(d)
                with open(file, 'w') as json_file:
                    json_file(data, json_file, indent=4)
            return True

        elif action is 'd':
            if os.path.isfile(file):
                with open(file, 'r') as json_file:
                    data = json.load(json_file)
                    data.pop(str(task.id))
                with open(file, 'w') as json_file:
                    json.dump(data, json_file)
            else:
                return False

    def get_task(self, id):
        if id is None:
            return None

        return self._tasks.get(id, None)

    def get_tasks(self):
        return self._tasks

    def run_task(self):
        pass

    def abort_task(self):
        pass

    def start(self):
        """
        Start the Scheduler
        Returns True on success
        """
        if self.running:
            return True

        gevent.spawn(self.main_thread)

        self.running = True
        return True

    def stop(self):
        """
        Stop the Scheduler
        Returns True if the shutdown was graceful
        """
        self.shutdown.set()
        wait_limit = 5
        start = time.time()
        while self.running:
            if time.time() - start > wait_limit:
                return False
            time.sleep(0.1)
        return True

    def main_thread(self):
        self.load_past_tasks()
        while not self.shutdown.is_set():
            for task in self._tasks:
                if task.status == Status.INIT:
                    task.status = Status.RUN
                    gevent.spawn_later()
                    gevent.queue

                if task.status == Status.WAIT:
                    task.status == Status.RUN
