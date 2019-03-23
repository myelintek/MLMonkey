import flask
import json
import uuid
from status import Status

# ref https://github.com/Alwayswithme/sysinfo
from model import sysinfo
from webapp import config
from webapp.scenario import Scenario, Task
from webapp.app import scheduler

blueprint = flask.Blueprint(__name__, __name__)


@blueprint.route('/', methods=['GET'])
def home():
    return 'homepage'


@blueprint.route('/hwinfo', methods=['GET', 'POST'])
def showHwInfo():
    if flask.request.method == 'POST':
        gpus = sysinfo.getGraphicsCardInfo()
        cpu = sysinfo.getCpuHwInfo()

        data = {
            'cpu': cpu,
            'memory': None,
            'disk': None,
            'gpu': gpus
        }
        with open('/data/hardware.json', 'w') as json_file:
            json.dump(data, json_file)

    with open('/data/hardware.json', 'r') as json_file:
        data = json.load(json_file)
    return flask.jsonify(data)


@blueprint.route('/scenario', methods=['POST'])
def createScenario():
    # Get data from the scenario form
    UID = str(uuid.uuid4())
    status = Status.INIT
    name = flask.request.form.get('name')
    data_dir = flask.request.form.get('data_dir')
    num_gpus = flask.request.form.get('num_gpus')
    batch_size = flask.request.form.get('batch_size')
    model = flask.request.form.get('model')
    variable_update = flask.request.form.get('variable_update')
    fp16 = flask.request.form.get('fp16')
    optimizer = flask.request.form.get('optimizer')
    data_format = flask.request.form.get('data_format')
    num_epochs = flask.request.form.get('num_epochs')
    scenario = Scenario(
        data_dir=data_dir,
        num_gpus=num_gpus,
        batch_size=batch_size,
        model=model,
        variable_update=variable_update,
        fp16=fp16,
        optimizer=optimizer,
        data_format=data_format,
        num_epochs=num_epochs
    )
    scheduler.add_task(Task(id=UID, name=name, scenario=scenario, status=status))
    '''
    data = []

    
    try:
        with open('/data/scenarios.json', 'r') as json_file:
            result = json.load(json_file)
            result.append(scenario)
            with open('/data/scenarios.json', 'w+') as json_file:
                json.dump(result, json_file, indent=4)
    except Exception as e:
        with open('/data/scenarios.json', 'w+') as json_file:
            data.append(scenario)
            json.dump(data, json_file, indent=4)
    '''

    return flask.jsonify({'id': UID})


@blueprint.route('/scenario/', methods=['GET'])
def showAllScenario():

    '''
    with open('/data/scenarios.json', 'r') as json_file:
        result = json.load(json_file)
    '''
    tasks = scheduler.get_tasks()
    response = []
    for task in tasks:
        response.append(task.json_dict())

    return flask.jsonify(response)


@blueprint.route('/scenario/result', methods=['GET'])
def showAllScenarioResult():
    tasks = scheduler.get_tasks()
    response = []
    for task in tasks:
        if task.status is Status.DONE:
            response.append(task.json_dict())
    return flask.jsonify(response)


@blueprint.route('/scenario/abort', methods=['POST'])
def abortAllScenario():
    pass


@blueprint.route('/scenario/run', methods=['POST'])
def runAllScenario():
    pass


@blueprint.route('/scenario/<job_id>', methods=['DELETE'])
def deleteScenario(job_id):
    '''
    with open('/data/scenarios.json', 'r') as json_file:
        scenarios = json.load(json_file)

    scenarios.pop(str(job_id))

    with open('/data/scenarios.json', 'w') as json_file:
        json.dump(scenarios, json_file)
    '''
    scheduler.delete_task(job_id)
    return True

@blueprint.route('/scenario/<job_id>', methods=['GET'])
def showScenario(job_id):
    '''
    response = {}
    with open('/data/scenarios.json', 'r') as json_file:
        scenarios = json.load(json_file)
        for scenario in scenarios:
            if job_id == scenario['UID']:
                response = scenario[job_id]
                break

    return flask.jsonify(response)
    '''
    task = scheduler.get_task(job_id)
    response = task.json_dict()
    return flask.jsonify(response)


@blueprint.route('/bandwidth', methods=['POST', 'GET'])
def runBandwidth():
    pass


@blueprint.route('/topology', methods=['POST', 'GET'])
def runTopology():
    # get folder or filepath by config

    try:
        with open('/data/topo.txt', 'r') as file:
            info = file.read()
    except:
        result = sysinfo.initTopology()

        with open('/data/topo.txt', 'w') as file:
            file.write(result)

    if flask.request.method == 'POST':
        result = sysinfo.initTopology()

        with open('/data/topo.txt', 'w') as file:
            file.write(result)
    else:
        with open('/data/topo.txt', 'r') as file:
            info = file.read()

    return info
