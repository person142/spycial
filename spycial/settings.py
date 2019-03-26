import os


def get_variable(name, default):
    value = os.environ.get(name)
    if value is None:
        return default

    return bool(int(value))


CACHE = get_variable('SPYCIAL_CACHE', True)
