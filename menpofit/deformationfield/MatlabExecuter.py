import subprocess
import os


class MatlabExecuter(object):
    def __init__(self):
        self.exe_str = '-nodisplay -nosplash -nodesktop -r "{};exit;"'
        self._dir = '~'

    def _run(self, command):
        os.chdir(self._dir)
        p = subprocess.Popen(['matlab', self.exe_str.format(command)])
        p.wait()

    def run_script(self, script):
        self._run('run({})'.format(script))

    def run_function(self, function):
        self._run(function)

    def cd(self, path):
        self._dir = path if path[0] == '/' else self._dir + path