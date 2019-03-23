from __future__ import absolute_import

import time


class Status():
    """
    A little class to store the state of Jobs and Tasks
    It's pickle-able!
    """

    # Enum-like attributes

    INIT = 'I'
    WAIT = 'W'
    RUN = 'R'
    DONE = 'D'
    ABORT = 'A'
    ERROR = 'E'

    def __init__(self, val):
        self.val = None
        self.name = ""
        self.set_dict(val)

    def __str__(self):
        return self.val

    # Pickling

    def __getstate__(self):
        return self.val

    def __setstate__(self, state):
        self.set_dict(state)

    # Operators

    def __eq__(self, other):
        if type(other) == type(self):
            return self.val == other.val
        elif type(other) == str:
            return self.val == other
        else:
            return False

    def __ne__(self, other):
        if type(other) == type(self):
            return self.val != other.val
        elif type(other) == str:
            return self.val != other
        else:
            return True

    # Member functions

    def set_dict(self, val):
        self.val = val
        if val == self.INIT:
            self.name = 'Initialized'
        elif val == self.WAIT:
            self.name = 'Waiting'
        elif val == self.RUN:
            self.name = 'Running'
        elif val == self.DONE:
            self.name = 'Done'
        elif val == self.ABORT:
            self.name = 'Aborted'
        elif val == self.ERROR:
            self.name = 'Error'
        else:
            self.name = '?'

    def is_running(self):
        return self.val in (self.INIT, self.WAIT, self.RUN)

    def is_ready(self):
        return self.val in (self.INIT, self.WAIT)

    def is_done(self):
        return self.val is self.DONE
