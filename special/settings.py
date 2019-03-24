import os


def get_variable(name, default):
    value = os.environ.get('SPECIAL_CACHE')
    if value is None:
        return default

    return bool(int(value))


CACHE = get_variable('SPECIAL_CACHE', True)
