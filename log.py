import os
import sys
from datetime import datetime


class Log:
    def __init__(self, _name, _params=None):
        self.time_started = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_dir = "saved_model/" + self.time_started

        self.terminal = sys.stdout
        self.log_folder = self.model_dir
        self.log_filename = _name
        if not os.path.exists(self.log_folder):
            print("NO LOG FOLDER FOUND - rebuiling it. ")
            os.makedirs(self.log_folder)
        self.full_filename = '{}/{}'.format(self.log_folder, self.log_filename)
        print('log file:{}'.format(self.full_filename))
        self.file_obj = open(self.full_filename, 'w+')
        self.write("{} \t\t Training params:\n".format(self.time_started))
        if _params is not None:
            for item in _params:
                self.write("{}: {}\n".format(item, _params[item]))

    def write(self, _str_to_write):
        self.terminal.write(_str_to_write)
        self.file_obj.write(_str_to_write)
        self.file_obj.flush()

    def close(self):
        time_ended = datetime.now().strftime("%Y/%m/%d-%H:%M.%S\n")
        self.write(time_ended)
        self.file_obj.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminal.close()
        self.file_obj.close()
