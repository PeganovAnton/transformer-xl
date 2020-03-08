# global variables shared between train.py, eval.py, ..., carrying info for a single user invocation-process pair
from typing import Optional

from data_preprocessing.bpe import GitBPE

event_writer = None
token_count = None
args = None
timeit_dict = None
logger = None

va_iter = None
te_iter = None
va_custom_iter = None


tie_projs = None
cutoffs = None
ntokens = None
device = None
state = None  # saveable state of optimization (model, optimizer, step, etc)

tokenizer: Optional[GitBPE] = None
