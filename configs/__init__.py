import re
from configs.configs import *


def get_config_by_name(name: str):
    assert re.match("\\w+", name), f"Config name should be a one word, but {name} was given"
    assert name in globals(), f"There is no configs called {name}"
    config = eval(name)
    return config
