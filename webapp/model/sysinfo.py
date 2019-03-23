import subprocess
import logging


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
    ex: ‘gpu’: [‘GeForce GTX 1080 Ti’, ‘GeForce GTX 1080 Ti’]
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
    ex: ‘cpu’: [‘Intel(R) Xeon(R) Silver 4116 CPU @ 2.10GHz’]
    :return: cpuinfo
    """
    cmd = "lscpu | grep 'Model name:' | sed -r 's/Model name:\s{1,}//g'"
    cpuinfo = _sh(cmd, True)
    cpuinfo = cpuinfo.strip()
    return cpuinfo


def get_mem_info():
    """
    detect information about memory.
    ex: ‘memory’: [‘Size: 32 GB, Type: DDR4, Speed: 2666 MHz, Manufacturer: Micron’]
    :return: meminfo
    """
    pass


def get_disk_info():
    """
    detect information about disk.
    ex: ‘disk’: [‘description: SCSI Disk,logical name: /dev/sda,size: 1787GiB (1919GB)’ ]
    :return: meminfo
    """
    pass


def init_bandwidth():
    pass


def init_topology():
    """
    run and return the nvidia topology.
    :return:
    """
    cmd = "nvidia-smi topo -m"
    topo = _sh(cmd, True)
    return topo
