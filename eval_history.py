import time
from datetime import datetime

class EvalHistoryFile:
    def __init__(self, prefix: str, filename: str=None) -> None:
        if filename is None:
            filename = "%s%i.csv" % (prefix, int(time.time()))

        self._filename = filename
        with open(self._filename, "w") as f:
            f.write("metric,label,time\n")

    def log(self, value: float, label: str) -> None:
        t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = "%f,%s,%s\n" % (value, label, t)
        with open(self._filename, "a") as f:
            f.write(line)
