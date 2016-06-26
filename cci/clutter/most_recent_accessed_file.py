import os

def program(files):
    max_mtime = 0
    for file in files:
        mtime = os.stat(file).st_mtime
        if mtime > max_mtime:
            max_mtime = mtime
            max_file = file

    print max_file