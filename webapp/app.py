from __future__ import absolute_import

import flask

app = flask.Flask(__name__)

import api.view
import scheduler

scheduler = scheduler.Scheduler()

app.register_blueprint(api.view.blueprint)

if __name__ == '__main__':
    app.debug = True
    app.run()