import numpy as np
import re

iteration_num = 5


def filter_time(file_path):
    with open(file_path, 'r') as fin:
        content = fin.read()
    reg = re.compile('real\t(\d+)m(.+?)s')
    results = reg.findall(content)
    if not results:
        print("empty")
        exit(1)

    ss = []
    for x in results:
        minute = int(x[0])
        second = float(x[1])
        second_all = 60 * minute + second
        ss.append(second_all)

    return np.mean(ss)

    

def work():
    original_time_file = 'time.txt'
    data_flow_time_file = 'time_data_flow.txt'
    value_pattern_time_file = 'time_value_pattern.txt'
    original_time = filter_time(original_time_file)
    data_flow_time = filter_time(data_flow_time_file)
    value_pattern_time = filter_time(value_pattern_time_file)
    overhead = data_flow_time / original_time + value_pattern_time / original_time
    print("%.2f" % overhead)

work()
