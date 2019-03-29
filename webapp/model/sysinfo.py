import subprocess
import logging
import re


def _sh(cmd, in_shell=False, get_str=True):
    """
    create a sub process and run command line with shell.
    :param cmd:
    :param in_shell:
    :param get_str:
    :return:
    """
    output = subprocess.check_output(cmd, shell=in_shell)
    if get_str:
        return str(output)
    return output


def get_graphics_card_info():
    """
    detect information about graphics card.
    ex: 'gpu': ['GeForce GTX 1080 Ti', 'GeForce GTX 1080 Ti']
    :return: graphics_info
    """
    cmd = "nvidia-smi -L"
    graphicsinfo = _sh(cmd, True)
    graphicsinfos = graphicsinfo.strip().split('\n')
    for index, info in enumerate(graphicsinfo):
        try:
            graphicsinfo[index] = str(graphicsinfo[:graphicsinfo.index('(') - 1])
        except Exception as e:
            logging.info(str(e))
    return graphicsinfos


def get_cpu_hwinfo():
    """
    detect information about cpu series.
    ex: 'cpu': ['Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz']
    :return: cpuinfo
    """
    cmd = "lscpu | grep 'Model name:' | sed -r 's/Model name:\s{1,}//g'"
    cpuinfo = _sh(cmd, True)
    cpuinfo = cpuinfo.strip()
    return cpuinfo


def get_mem_info():
    """
    detect information about memory.
    ex: 'memory': ['Size: 32 GB, Type: DDR4, Speed: 2666 MHz, Manufacturer: Micron']
    :return: meminfo
    """
    cmd = "dmidecode -t memory"
    content = _sh(cmd, True)
    parsing = False
    splitter = ': '
    attrs = ['Size', 'Type', 'Speed', 'Manufacturer', 'Locator']
    mem_list = []
    data = content.split('\n')
    for i in data:
        line = i.strip()
        if not parsing and line == 'Memory Device':
            parsing = True
            mem = {}
        if parsing and splitter in line:
            (key, value) = line.split(splitter, 1)
            if key in attrs:
                mem[key] = value

        # read a empty, end the parsing
        elif parsing and not line:
            parsing = False
            mem_list.append(mem)
    return mem_list


def get_disk_info():
    """
    detect information about disk.
    ex: 'disk': ['description: SCSI Disk,logical name: /dev/sda,size: 1787GiB (1919GB)' ]
    :return: meminfo
    """

    def disk_list():
        """
        find out how many disk in this machine
        """
        sds = _sh('ls -1d /dev/sd[a-z]', in_shell=True)
        sd_list = [x for x in sds.split('\n') if x]
        return sd_list

    def countSize(disks):
        sum = 0
        for i in disks:
            cmd = 'blockdev --getsize64 ' + i
            output = _sh(cmd, True)
            sum += int(output) // (10 ** 9)

        return "{} GB".format(sum)

    disks = disk_list()
    cmd = ['smartctl', '-i']
    parsing = False
    splitter = ':'
    disk_list = []
    for i in disks:
        new_cmd = cmd[:]
        new_cmd.append(i)
        p = subprocess.Popen(new_cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)
        for content in p.stdout:
            line = content.strip()
            if not parsing and 'START OF INFORMATION' in line:
                parsing = True
                disk = {}
            if parsing and splitter in line:
                key, value = line.split(splitter, 1)
                value = value.strip()
                if key in 'Model Family':
                    disk['model'] = value
                elif key in 'Device Model':
                    disk['device'] = value
                elif key in 'User Capacity':
                    p = re.compile('\[.*\]')
                    m = p.search(value)
                    disk['capacity'] = m.group()
            elif parsing and not line:
                parsing = False
                disk['node'] = i
                disk_list.append(disk)
        disk_info = {'countSize': countSize(disks), 'disk_list': disk_list}
    return disk_info


def init_bandwidth():
    """

    :return:
    """
    executor = 'bash'
    exec_path = '/usr/local/cuda/samples/1_Utilities'
    exec_file = 'bandwidthTest'
    cmd = [executor, exec_path, exec_file]
    result = _sh(cmd, True)
    return result


def init_topology():
    """
    run and return the nvidia topology.
    :return:
    """
    cmd = "nvidia-smi topo -m"
    topo = _sh(cmd, True)
    return topo
