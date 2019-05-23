import datetime
import os
import sys

import torch
import torch.distributed as dist

import globals as g


def toscalar(t):  # use on python scalars/pytorch scalars
    """Converts Python scalar or PyTorch tensor to Python scalar"""
    if isinstance(t, (float, int)):
        return t
    if hasattr(t, 'float'):
        t = t.float()   # half not supported on CPU
    if hasattr(t, 'item'):
        return t.item()
    else:
        assert len(t) == 0
        return t[0]


def _info(_type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback
        import pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb)  # more "modern"


def pdb_on_error():
    # todo(y): doesn't work when called from other files?
    sys.excepthook = _info


def get_world_size() -> int:
    return int(os.environ.get('WORLD_SIZE', 1))


def get_global_rank() -> int:
    """Returns global rank (from env), or 0 if not set"""
    return int(os.environ.get('RANK', 0))


def one_of(l):
    assert len(l) == 2
    if l[0]:
        return l[0]
    elif l[1]:
        return l[1]
    else:
        assert f"List {l} has more than one non-zero entries"
    

def dist_sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


# no_op method/object that accept every signature
class NoOp:
    def __getattr__(self, *_args):
        def no_op(*_args, **_kwargs): pass

        return no_op


# Deprecated method, regular restore + DDP already broadcasts args
# def dist_restore_from_checkpoint(ddp_model, checkpoint_fn: str, force_fp16=False):
#     """Restores model wrapped in DistributedDataParallel from checkpoint file. Assumes checkpoint was saved
#     as torch.save(ddp.module) or distributed_save_checkpoint
#     """

#     if get_global_rank() == 0:
#         saved_model = torch.load(checkpoint_fn)
#         state_dict = saved_model.state_dict()
#         if force_fp16:
#             for name in state_dict:
#                 state_dict[name] = state_dict[name].half()
#         ddp_model.module.load_state_dict(state_dict)

#     pp = next(ddp_model.module.parameters())
#     print(f"{get_global_rank()}  -- Before broadcast {pp.view(-1)[0]}")
#     for p in ddp_model.module.parameters():
#         if torch.is_tensor(p):
#             dist.broadcast(p, 0)
#     print(f"{get_global_rank()}  -- After broadcast {pp.view(-1)[0]}")


def restore_from_checkpoint(model, optimizer=None, checkpoint_fn: str = '',
                            optimizer_state_dict_fn: str = '', force_fp16=False,
                            override_lr=0):
    """Restores model wrapped in DistributedDataParallel from checkpoint file.
    Assumes checkpoint was saved as torch.save(ddp.module).

    If optimizer_state_dict_fn is provided, also tries to restore optimizer state from state_dict saved in that file.

    Assumes optimizer is regular optimizer, not FP16Optimizer(optimizer), must wrap FP16 on top
    of restored optimizer here.
    """

    saved_model = torch.load(checkpoint_fn)
    state_dict = saved_model.state_dict()
    if force_fp16:
        for name in state_dict:
            state_dict[name] = state_dict[name].half()
    model.load_state_dict(state_dict)

    assert 'FP16_Optimizer' not in type(optimizer).__name__, f"Checkpoint restore works on PyTorch optimizers, but " \
        f"found {type(optimizer).__name__}, found unwrap your optimizer first"
    if optimizer_state_dict_fn:
        optimizer_state_dict = torch.load(optimizer_state_dict_fn)
        # another layer of indirection added for FP16Optimizer
        if 'optimizer_state_dict' in optimizer_state_dict:
            optimizer_state_dict = optimizer_state_dict['optimizer_state_dict']
        if override_lr:
            optimizer_state_dict['param_groups'][0]['lr'] = override_lr
        optimizer.load_state_dict(optimizer_state_dict)
    

def dist_save_checkpoint(ddp_model, optimizer_, directory: str, suffix=''):
    """Saves model/optimizer into {directory}/optimizer-{suffix}.py and {directory}/model-{suffix}.pt"""
    if get_global_rank() != 0:
        return
    with open(directory + f'/model-{suffix}.pt', 'wb') as f_1:
        torch.save(ddp_model.module, f_1)
    with open(directory + f'/optimizer-{suffix}.pt', 'wb') as f_1:
        torch.save(optimizer_.state_dict(), f_1)


def save_state(state, fn):
    """Saves"""
    if get_global_rank() != 0:
        return
    if state.model.__class__.__name__ == 'DistributedDataParallel':
        state.model = state.model.module
    torch.save(state, fn)


def load_state(fn):
    """Loads state from fn"""
    if get_global_rank() != 0:
        return
    return torch.load(fn)


def cancel_shutdown():
    args = g.args
    if args.local:
        return
    if args.local_rank > 0:
        return
    os.system('shutdown -c')


def current_timestamp(timezone: str = 'America/Los_Angeles') -> str:
    """Gives timestamp formated like 2019-04-15_11-29-51. correct to local timezone (PDT) if running on AWS (which is UTC)"""
    pacific_tz = pytz.timezone(timezone)
    localtime = pytz.utc.localize(datetime.datetime.now(), is_dst=None).astimezone(pacific_tz)
    return localtime.strftime('%Y-%m-%d_%H-%M-%S')


def assert_close(observed, target, rtol=1e-5, atol=1e-3):
    relative = abs(target-observed)/target
    assert relative < rtol, f"rtol {rtol} exceeded at {relative}, observed={observed}, target={target}"

    absolute = abs(target-observed)
    assert absolute < rtol, f"atol {atol} exceeded at {absolute}, observed={observed}, target={target}"
