import subprocess


def pipe_read(command):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout


def cleanup(arch):
    pipe_read(['make', 'clean'])
    if arch is not None:
        pipe_read(['make', 'ARCH={}'.join(arch)])
    else:
        pipe_read(['make'])
