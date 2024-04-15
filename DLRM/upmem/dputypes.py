from ctypes import *
import csv
from enum import IntEnum

## ctypes ##

class CtypesEnum(IntEnum):
    """ctypes semi-compatible IntEnum superclass."""
    @classmethod
    def from_param(cls, obj):
        return int(obj)

class DpuRuntimeConfigEnum(CtypesEnum):
    """ctypes compatible dpu_runtime_config struct"""
    RT_ALL = 0
    RT_LAUNCH = 1

class DpuTimespec(Structure):
    """ctypes compatible dpu_timespec struct"""
    _fields_ = [
        ("tv_nsec", c_long),
        ("tv_sec", c_long)
    ]

    def __init__(self, tv_sec=0, tv_nsec=0):
        super(DpuTimespec, self).__init__(tv_nsec, tv_sec)

    def as_ms(self, offset=0):
        return ((self.tv_sec * 1000) + (self.tv_nsec / 1E+6)) - offset

    def __repr__(self):
        return f"(tv_sec={self.tv_sec}, tv_nsec={self.tv_nsec} (as_ms={self.as_ms()}))"

class DpuRuntimeInterval(Structure):
    """ctypes compatible dpu_runtime_interval struct"""
    _fields_ = [
        ("start", DpuTimespec),
        ("stop", DpuTimespec)
    ]

    def __init__(self):
        super(DpuRuntimeInterval, self).__init__(DpuTimespec(), DpuTimespec())

    def __repr__(self):
        return f"(start={self.start}, stop={self.stop})"

class DpuRuntimeGroup(Structure):
    """ctypes compatible dpu_runtime_group struct"""
    _fields_ = [
        ('in_use', c_uint),
        ('length', c_uint),
        # Array of dpu_runtime_interval
        ('intervals', POINTER(DpuRuntimeInterval))
    ]

    def __init__(self, in_use=0, length=4, values=None):
        if values is None:
            v = (DpuRuntimeInterval * length)()
        else:
            v = values
        super(DpuRuntimeGroup, self).__init__(in_use, length, v)


    def __repr__(self):
        return f"(in_use={self.in_use}, length={self.length}, intervals={self.intervals})"

class DpuRuntimeTotals(Structure):
    """ctypes compatible dpu_runtime_interval struct"""
    _fields_ = [
        ("execution_time_prepare", c_double),
        ("execution_time_populate_copy_in", c_double),
        ("execution_time_copy_in", c_double),
        ("execution_time_copy_out", c_double),
        ("execution_time_aggregate_result", c_double),
        ("execution_time_launch", c_double)
    ]

    def __init__(self):
        super(DpuRuntimeTotals, self).__init__(0, 0, 0, 0, 0, 0)

## Helper Functions ##

def customresize(array, new_size):
    resize(array, sizeof(array._type_)*new_size)
    return (array._type_*new_size).from_address(addressof(array))

def write_results(config, runtime_group, headers=["DPU", "Start", "Stop"]):
    if not config.runtimes:
        return

    with open(config.runtime_file, 'a') as csvfile:
        rtwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if headers is not None and len(headers) > 0:
            rtwriter.writerow(headers)
        for nr, item in enumerate(runtime_group):
            for i in range(0, item.in_use):
                rtwriter.writerow([nr, f'{item.intervals[i].start.as_ms():.16f}', f'{item.intervals[i].stop.as_ms():.16f}'])
