from collections import namedtuple
import subprocess
import os
import sys

TestCase = namedtuple('TestCase', ['path', 'command', 'options'])


def setup():
    test_cases = []
    # unit case
    # real cases

    return test_cases


def pipe_read(command):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout


def cleanup():
    pipe_read(['make', 'clean'])
    pipe_read(['make'])
    pipe_read(['rm', '-rf', 'gvprof-measurements*'])
    pipe_read(['rm', '-rf', 'gvprof-database*'])


def test(test_cases, bench):
    for test_case in test_cases:
        if bench is not None and bench != test_case.path:
            continue

        os.chdir(test_case.path)

        cleanup()

        os.chdir('../..')


bench = None
if len(sys.argv) > 1:
    bench = str(sys.argv[1])

test(setup(), bench)