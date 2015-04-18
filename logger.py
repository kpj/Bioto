VERBOSE = True

def log(*msg, **kwargs):
    if VERBOSE:
        print(*msg, **kwargs)
