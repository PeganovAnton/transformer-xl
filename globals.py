from log import FileLogger

# global variables shared between train.py, eval.py, ..., carrying info for a single user invocation-process pair
event_writer = None
token_count = None
args = None
timeit_dict = None
logger: FileLogger = None

corpus = None
va_iter = None
te_iter = None


tie_projs = None
cutoffs = None
ntokens = None
device = None
state = None  # saveable state of optimization (model, optimizer, step, etc)

last_save_timestamp = None
