import re
import csv
import os
import traceback
import json
from os import listdir
from os.path import join, isfile
from abc import ABCMeta, abstractmethod
from datetime import datetime, timezone

LOGS_PATH = os.path.join("/workspace", "logs")
pattern = re.compile("(\d*.\d*) real")
log_groups = {}

IMAGE_CLASSIFICATION="image_classification"
RNN_TRANSLATOR="rnn_translator"
OBJECT_DETECTION="object_detection"
SPEECH_RECOGNITION="speech_recognition"


#ALL_CASES=(IMAGE_CLASSIFICATION, RNN_TRANSLATOR,
#           OBJECT_DETECTION, SPEECH_RECOGNITION)
ALL_CASES=(IMAGE_CLASSIFICATION, RNN_TRANSLATOR,
           OBJECT_DETECTION)


NONE='None'


class Parser_Base(metaclass=ABCMeta):

    def __init__(self, root_dir, case=''):
        self.root_dir = root_dir
        self.case = case
        self.log_dir = os.path.join(root_dir, case)
        self.results = {}
        self.pattern_rstart=':::MLPv0\.5\.0.* (\d+.\d+) \(.*\) .*run_start'
        self.pattern_rend=':::MLPv0\.5\.0.* (\d+.\d+) \(.*\) .*run_final'
        self.pattern_accuracy=':::MLPv0\.5\.0.*eval_accuracy: {(.*)}'
        self.pattern_target=':::MLPv0\.5\.0.*eval_target: {(.*)}'
        self.pattern_cmd='python .*'

    def _readlines_reverse(self, filename):
        with open(filename) as qfile:
            qfile.seek(0, os.SEEK_END)
            position = qfile.tell()
            line = ''
            while position >= 0:
                qfile.seek(position)
                try:
                    next_char = qfile.read(1)
                except:
                    # swallow UnicodeDecodeError and others
                    next_char = ''
                # reverse read char by char until the \n symbol
                # and yeild the line to caller
                if next_char == "\n":
                    yield line[::-1]
                    line = ''
                else:
                    line += next_char
                position -= 1
            yield line[::-1]

    def _readline_tail(self, log, pattern, max_lines=500):
        for i, line in (enumerate(self._readlines_reverse(log))):
            # match pattern
            tmp = re.findall(pattern, line)
            if tmp or i > max_lines:
                return tmp

    def _readline_head(self, log, pattern, max_lines=500):
        with open(log) as f:
            for i, line in enumerate(f):
                try:
                    tmp = re.findall(pattern, line)
                    if tmp or i > max_lines: # found or meet a boundary
                        return tmp
                except:
                    # swallow
                    line = ''

    def parse_duration(self, log, pattern_start, pattern_end):
        result={}
        # find the index of :::MLPv0.5.0 resnet ... run_start
        tmp = self._readline_head(log, pattern_start)
        result.update({"run_start": tmp[0] if tmp else NONE})

        # find the index of :::MLPv0.5.0 resnet ... run_final
        tmp = self._readline_tail(log, pattern_end)
        if not tmp: raise (Exception('Error: Log has no run_final!'))

        result.update({"run_final": tmp[0] if tmp else NONE})

        # calculate the duration
        if result and result["run_final"] != NONE and result["run_start"] != NONE:
            # get duration
            duration = float(result["run_final"]) - float(result["run_start"])
            sutc = datetime.fromtimestamp(int(float(result["run_start"])))
            futc = datetime.fromtimestamp(int(float(result["run_final"])))
            duration_utc = str(futc-sutc)
            sutc = sutc.isoformat()
            futc = futc.isoformat()
            print("run_duration = {}, in datetime = {}".format(duration, duration_utc))
        else:
            duration = NONE
            sutc = NONE
            futc = NONE
            duration_utc = NONE

        result.update({"run_duration": duration,
                       "run_start_utc": sutc,
                       "run_final_utc": futc,
                       "duration_utc": duration_utc})

        return result

    @abstractmethod
    def parse_log(self):
        return True

    # get accuracy and target
    def parse_criteria(self, log, pattern_accuracy, pattern_target):
        result={}
        tmp = self._readline_tail(log, pattern_accuracy)
        result.update({"accuracy": tmp if tmp else NONE})
        tmp = self._readline_tail(log, pattern_target)
        result.update({"target": tmp if tmp else NONE})
        return result

    def parse_executed_cmd(self, log,
                           pattern_cmd='python .*'):
        result={}
        tmp = self._readline_head(log, pattern_cmd)
        print("pattern_cmd = {}".format(pattern_cmd))
        result.update({"cmd": tmp if tmp else NONE})
        return result


class Parser(Parser_Base):

    # if you want to add additional strings, implement this function
    # in inheritance.
    def parse_additional(self, log):
        return {}

    def _find_next_log_entry(self, root_dir):
        # context manager of scandir supported only 3.6 above
        #with os.scandir(root_dir) as it:
        try:
            for entry in os.scandir(root_dir):
                if entry.is_dir():
                    #print("is_dir() = {}".format(entry.path))
                    yield from self._find_next_log_entry(str(entry.path))
                elif entry.is_file():
                    #print("is_file() = {}".format(entry.path))
                    yield entry
                else:
                    print("warning: unknown object: {}".format(entry))
        except OSError:
            print('Cannot access ' + root_dir +'. Probably a permissions error')

    def parse_log(self):
        print("\n----- parse case {} -----".format(self.case))
        # try to iterate to file
        for entry in self._find_next_log_entry(self.log_dir):
            log = entry.path
            log_name = re.sub('^'+self.log_dir, '', entry.path)
            print("\n*parse log: {} in {}".format(log_name, log))
            try:
                # get duration
                result = self.parse_duration(log,
                                             self.pattern_rstart,
                                             self.pattern_rend)
                # get criteria
                criteria = self.parse_criteria(log,
                                               self.pattern_accuracy,
                                               self.pattern_target)
                result.update(criteria)

                # get command
                cmd = self.parse_executed_cmd(log, self.pattern_cmd)
                result.update(cmd)

                # additional
                others = self.parse_additional(log)
                result.update(others)

                self.results.update({log_name:result})

            except Exception as e:
                print("case {}, exception: {}".format(self.case, e))
                traceback.print_exc()
        return self.results


class Parser_RTR(Parser):

    def __init__(self, root_dir, case=''):
        super().__init__(root_dir, case)
        self.pattern_cmd = '0: Run arguments: .*'
        self.pattern_target=':::MLPv0\.5\.0.*eval_target: (.*)'

    def parse_additional(self, log):
        pattern = "Performance: (.*)"
        result = {}
        tmp = self._readline_tail(log, pattern)
        result.update({"performance": tmp if tmp else NONE})
        return result


class Parser_IMC(Parser):

    def __init__(self, root_dir, case=''):
        super().__init__(root_dir, case)
        #self.pattern_accuracy = ''
        #self.pattern_target=''

    def parse_duration(self, log, pattern_start, pattern_end):
        result = {}
        pattern = "(\d*.\d*) real"
        tmp = self._readline_tail(log, pattern)
        result.update({"real": tmp if tmp else NONE})
        return result

    # no command found, read the parameter section instead.
    def parse_executed_cmd(self, log, pattern_cmd=""):
        result = {}
        total_lines = 14 # lines to be read
        max_lines = 500
        with open(log) as f:
            for i, line in enumerate(f):
                try:
                    tmp = re.findall("TensorFlow: .*", line)
                    if tmp or i > max_lines: # found or meet a boundary
                        cmd_str = tmp[0]+"\n"
                        for i in range(total_lines):
                            cmd_str += f.readline()
                        #print("vis dev: cmd_str = {}".format(cmd_str))
                        result.update({"cmd":cmd_str})
                        break
                except:
                    # swallow
                    line = ''
        return result

    def parse_additional(self, log):
        pattern = "total images/sec: (\d+.\d+)"
        result = {}
        tmp = self._readline_tail(log, pattern)
        result.update({"images/sec": tmp if tmp else NONE})
        return result


# log messages are refered for parsing object_detection
    # :::MLPv0.5.0 maskrcnn 1559084034.883194447 (/workspace/object_detection/maskrcnn_benchmark/utils/mlperf_logger.py:40) eval_accuracy: {"epoch": 13, "value": {"BBOX": 0.38108277320861816, "SEGM": 0.3452053368091583}}
    # :::MLPv0.5.0 maskrcnn 1559084034.887276888 (/workspace/object_detection/maskrcnn_benchmark/utils/mlperf_logger.py:40) run_stop: {"success": true}
    # :::MLPv0.5.0 maskrcnn 1559084034.880022526 (/workspace/object_detection/maskrcnn_benchmark/utils/mlperf_logger.py:40) eval_target: {"BBOX": 0.377, "SEGM": 0.339}

# log messages are refered for parsing rnn_translator
    # find the index of Performance: Epoch: 5        Training: 47413 Tok/s   Validation: 155072 Tok/s
    # :::MLPv0.5.0 gnmt 1559234658.132781744 (train.py:495) eval_accuracy: {"epoch": 5, "value": 24.04}
    # :::MLPv0.5.0 gnmt 1559234658.133461475 (train.py:497) eval_target: 24.0


# The parser will parse the cases in ALL_CASES under LOGS_PATH,
# you can specify another log path by pass in the parameter and
# case name.
# The output data structure is:
# {
#    case_name1:
#        logname1: {
#            results1{}
#        }
#        logname2: {
#            results2{}
#        }
#    case_name2: ...
#
#for example:
#    rnn_translator:{
#        {'final-benchmark-2019-05-30-18':
#           {'accuracy': ['"epoch": 5, "value": 24.04'],
#            'run_final_utc': '2019-05-30T16:44:18',
#            'run_start_utc': '2019-05-30T10:34:18'
#            'duration_utc': datetime.timedelta(0, 22200),
#            'target': ['24.0'],
#            'run_duration': 22199.631192922592,
#            'performance': ['Epoch: 5\tTraining: 47413 Tok/s\tValidation: 155072 Tok/s'],
#            'run_final': '1559234658.135656118',
#            'run_start': '1559212458.504463196',
#            'cmd': "cmd"
#           }
#         'log2':
#           {
#             ...
#           }
#        }
#    object_detection:{
#        ...
#    }
#  }
def main():

    print("all cases: {}".format(ALL_CASES))

    all_results = {}
    for case in ALL_CASES:
        if case == RNN_TRANSLATOR:
            parser = Parser_RTR(LOGS_PATH, RNN_TRANSLATOR)
        elif case == IMAGE_CLASSIFICATION:
            parser = Parser_IMC(LOGS_PATH, case)
        else:
            parser = Parser(LOGS_PATH, case)

        try:
            result=parser.parse_log()
            all_results.update({case:result})
        except Exception as e:
            print("case {}, exception: {}".format(case, e))
            traceback.print_exc()

    print(json.dumps(all_results, indent=4))

if __name__ == '__main__':
    main()


