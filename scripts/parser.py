import re
import csv
import os
from os import listdir
from os.path import join, isfile
LOGS_PATH = "/workspace/logs"
pattern = re.compile("(\d*.\d*) real")
log_groups = {}

def readlines_reverse(filename):
    with open(filename) as qfile:
        qfile.seek(0, os.SEEK_END)
        position = qfile.tell()
        line = ''
        while position >= 0:
            qfile.seek(position)
            next_char = qfile.read(1)
            if next_char == "\n":
                yield line[::-1]
                line = ''
            else:
                line += next_char
            position -= 1
        yield line[::-1]

dirs = listdir(LOGS_PATH)
for d in dirs:
    files = [f for f in listdir(join(LOGS_PATH, d)) if isfile(join(LOGS_PATH, d, f))]
#    files = sorted(files, key=lambda x:x[0])
    result_time = {}
    
    for filename in files:
#        print(filename[0])
        i = 0
        time = 0 
        times = []
        for line in readlines_reverse(join(LOGS_PATH, d, filename)):
            times += pattern.findall(line) 
            i=i+1
            if i >=10: break
        if times: time = max(times)
        filename = filename[:-4]
        result_time[filename] = time

    log_groups[d] = result_time 
#print(log_groups)
for key1 in log_groups.keys():
    with open('/web/csv/%s.csv'%(key1), 'w') as f:
        print(sorted(log_groups[key1].keys(), key=lambda x:x[0]))
        for key2 in sorted(log_groups[key1].keys(), key=lambda x:x[0]):
            f.write("%s,%s\n"%(key2,log_groups[key1][key2]))