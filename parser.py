import re
import csv
import os
LOGS_PATH = "/workspace/logs"
pattern = re.compile("(\d*.\d*) real")
result_time = {}

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

files = [f for f in os.listdir(LOGS_PATH) if os.path.isfile(os.path.join(LOGS_PATH, f))]
for filename in files:
    i=0
    times = []
    for line in readlines_reverse(os.path.join(LOGS_PATH,filename)):
        times += pattern.findall(line) 
        i=i+1
        if i >=10: break
    time = max(times)
    result_time[filename] = time
print(result_time)
