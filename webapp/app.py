from __future__ import absolute_import

import flask

app = flask.Flask(__name__)

import views.benchmark
import scheduler

scheduler = scheduler.Scheduler()

app.register_blueprint(views.benchmark.blueprint)

if __name__ == '__main__':
    app.debug = True
    app.run()