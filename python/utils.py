import subprocess
import csv


def pipe_read(command, debug=False):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if debug is True:
        print(stdout)
    return stdout


def cleanup(arch):
    pipe_read(['make', 'clean'])
    if arch is not None:
        pipe_read(['make', 'GPU_ARCH="-arch {}"'.join(arch)])
    else:
        pipe_read(['make'])


def nsys_profile(command, kernels):
    pipe_read(['nsys', 'profile', '-f', 'true', '-o', 'tmp'] + command)
    pipe_read(['nsys', 'stats', '--report', 'gpukernsum', '--report', 'gpumemtimesum',
               '--format', 'csv', '-o', 'tmp', '--force-overwrite', './tmp.qdrep'])

    kernel_times = dict()

    gpu_kernel_time = 0.0

    with open('tmp_gpukernsum.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        first_row = True
        for row in spamreader:
            if first_row is True:
                first_row = False
                continue

            time = row[1]
            kernel_args_name = row[6].replace('"', '').replace('void ', '')
            gpu_kernel_time += float(time)

            for kernel_name, template in kernels:
                if template is True:
                    match_kernel_name = kernel_name
                else:
                    match_kernel_name = kernel_name + '('
                if kernel_args_name.startswith(match_kernel_name) is True:
                    kernel_times[kernel_name] = float(time)
                    break

    gpu_mem_time = 0.0

    with open('tmp_gpumemtimesum.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        first_row = True
        for row in spamreader:
            if first_row is True:
                first_row = False
            else:
                gpu_mem_time += float(row[1])

    return kernel_times, gpu_kernel_time, gpu_mem_time
