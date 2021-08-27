import time
import csv
import collections

from datetime import datetime
from pathlib import Path


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def read_csv_to_dict(path):
    columns = collections.defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.items():
                columns[k].append(v)
    return columns


def write_to_csv(path, header, *args):
    ensure_file_dir(path)
    if path.exists():
        aw = "a"
        wrt_header = False
    else:
        aw = "w"
        wrt_header = True
    with open(path, aw, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if wrt_header:
            writer.writerow(header)
        for values in args:
            writer.writerow(values)


def timeit(f):
    def wrapper(*args, **kwargs):
        log("Started:", f.__qualname__)
        t = time.time()
        res = f(*args, **kwargs)
        log(f"Finished: {f.__qualname__} elapsed: {time.time() - t:.2f}s")
        return res

    return wrapper


def safe_create_dir(d: Path):
    """
    Uses new pathlib
    Parameters
    ----------
    d: :obj:`pathlib.Path`
    """
    if not d.exists():
        log("Directory is not found, creating:", d)
        d.mkdir(parents=True)


def ensure_file_dir(file_path: Path):
    """
    Uses new pathlib
    Parameters
    ----------
    file_path: :obj:`pathlib.Path`
    """
    safe_create_dir(file_path.parent)


log_f = None
log_p = None


def change_log_path(path):
    global log_p, log_f
    if log_p == path:
        return
    if log_f:
        log_f.close()
    log_p = path
    ensure_file_dir(path)
    log_f = open(path, "a")
    log("Initialized log_path:", path)


def logr(*args, **kwargs):
    log(*args, **kwargs, end="\r")


def log(*args, **kwargs):
    ts = datetime.now().strftime(ISO_FORMAT)[:-3]
    if "ts" not in kwargs or kwargs["ts"] is not False:
        args = [ts, *args]
    if "ts" in kwargs:
        del kwargs["ts"]
    print(*args, **kwargs, flush=True)
    if log_f:
        print(*args, **kwargs, file=log_f)
        log_f.flush()
