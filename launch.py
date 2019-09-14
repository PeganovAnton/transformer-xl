#!/usr/bin/env python
"""Train transformer-XL on AWS."""

import argparse
import copy
import pprint

from attrdict import AttrDict
import re
import util

import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='',
                    help='which training config to use')
parser.add_argument('--spot', action='store_true',
                    help='Use cheaper spot instances')
parser.add_argument('--skip_setup', type=int, default=0,
                    help='Make startup slightly faster by skiping various initialization tasks, like '
                         'tmux/efs setup. Only use on reruns.')

# debug flags
parser.add_argument('--skip_wandb', type=int, default=0,
                    help='turns off wandb logging for this run')
parser.add_argument('--log_all_workers', type=int, default=0,
                    help='forces each process to produce separate log')

########
# Deprecated flags, everything else should be specified as part of config
#######
parser.add_argument('--wiki', action='store_true',
                    help='Train on all of wikipedia.')
parser.add_argument('--git', action='store_true',
                    help='Train on git dataset.')
parser.add_argument('--bpe', action='store_true',
                    help='Use BPE to reduce vocab instead of adaptive softmax div')

parser.add_argument('--num_rings', type=int, default=16)

parser.add_argument('--machines', type=int, default=0,
                    help="how many machines to use")
parser.add_argument('--instance_type', type=str, default='',
                    help="how many machines to use")
parser.add_argument('--checkpoint_each_epoch', type=int, default=0,
                    help='whether to save checkpoint at each epoch')
parser.add_argument('--checkpoint', type=str, default='',
                    help='restore from this checkpoint')
parser.add_argument('--checkpoint_state', type=str, default='',
                    help='restore from this state checkpoint')

args = parser.parse_args()

# batch size against which batch-normalized is measured. IE, base_lr=123 means batch size B gets lr 123*(B/BASE_LR_BATCHSIZE)
BASE_LR_BATCHSIZE = 32


################################################################################
# train.py parameter sets
################################################################################

# these are common values of parameters for train.py

# Match https://github.com/kimiyoung/transformer-xl/blob/master/tf/scripts/wt103_large_tpu.sh
# 300 million transformer-xl architecture
TRANSFORMER_360 = {
    'n_layer': 18,
    'd_model': 1024,
    'n_head': 16,
    'd_head': 64,
    'd_inner': 4096,
    'dropout': 0.2,
    'dropatt': 0.2,
    'optim': 'lamb',
    'warmup_tokens': 0,
    'tgt_len': 384,
    'mem_len': 384,
    'eval_tgt_len': 128,
    'fp16': True,
    'dynamic_loss_scale': True,
    'init_std': 0.005,
    'div_val': 4,
}

# 100 million transformer-xl architecture
# Divisible by 8 for fp16 compatibility.
TRANSFORMER_110 = {
    'n_layer': 16,
    'd_model': 512,
    'n_head': 8,
    'd_head': 48,
    'd_inner': 2048,
    'dropout': 0.1,
    'dropatt': 0.0,
    'optim': 'lamb',
    'tgt_len': 128,
    'mem_len': 128,
    'eval_tgt_len': 128,
}

# dataset specific parametesr
WIKI_DATASET = {
    'data': 'data/wikiextracted',
    'dataset': 'wiki',
    'dropatt': 0.1,
    'dropout': 0.1,
    'div_val': 1,
    'adaptive': False,
}
GIT_DATASET = {
    'data': 'data/git',
    'dataset': 'git',
    'div_val': 1,
    'bpe': True,
    'adaptive': False,
}

################################################################################
# Launcher configs
# Each config is a complete specification of training run

# default values inherited by all configs
root_config = {
    'conda_env': 'pytorch_p36'
}

basic = {
    'name': 'basic',  # determines name of machine and run in logging
    'machines': 1,
    'image_name': 'cybertronai01_git02',

    # batch-size and learning rate are given as top level params because they are used to set local learning
    # rates for individual train.py scripts
    'base_lr': 0.001 / 4,
    'instance_type': 'p3.16xlarge',
    'local_batch_size': 6,

    # These are additional parameters passed to train.py script, in addtion to "root_worker_params"
    # If list of dicts is given, they are merged, with later values overwriting earlier values
    'worker_params': [
        TRANSFORMER_360,
        GIT_DATASET,
        {
            'checkpoint': "https://s3.amazonaws.com/yaroslavvb2/data/git360-84-model.pt",  # 84% accuracy checkpoint
            'fp16': True,
            'warmup_tokens': 50e5,
            'dynamic_loss_scale': True,
            'scheduler': 'constant',
            'data': 'data/git',
            'dataset': 'git',
        }
    ]
}


def _get_nccl_params():
    params = f'NCCL_DEBUG=VERSION '

    params += f'NCCL_MIN_NRINGS={args.num_rings} ' \
              f'NCCL_MAX_NRINGS={args.num_rings} '
    return params


def main():
    global root_config

    assert not args.instance_type, "specify instance_type as part of config"
    assert not args.machines, "specify number of machines as part of config"
    assert re.match('\\w+', args.config)
    assert args.config in globals(), f'no config called {args.config}'

    config = copy.copy(root_config)
    config.update(eval(args.config))
    config = AttrDict(config)  # easier access to dictionary entries

    instance_info = ncluster.aws_backend.INSTANCE_INFO[config.instance_type]
    num_gpus_per_machine = instance_info['gpus']

    job = ncluster.make_job(name=config.name,
                            run_name=f"{config.name}",
                            num_tasks=config.machines,
                            image_name=config.image_name,
                            instance_type=config.instance_type,
                            spot=args.spot,
                            skip_setup=args.skip_setup)

    job.rsync('.')
    if not args.skip_setup:
        job.run(f'killall python || echo failed && '  # kill previous run
                f'source activate {config.conda_env} && ' +
                # protobuf https://github.com/tensorflow/models/issues/3995
                f'pip uninstall -y sagemaker && ' +   # sagemaker pins many libraries to incompatible old versions,
                f'pip uninstall -y protobuf && ' +
                f'pip install -U protobuf && ' +
                f'pip install -r requirements.txt --ignore-installed wrapt')
    # job.run('bash get_git_data.sh')
    # job.run('bash get_git_data_85gb.sh')
    # job.run('curl "https://github-lm.s3.amazonaws.com/github-projects_p3dn-2d_best.pt" '
    #         '-o github-projects_p3dn-2d_best.pt')
    else:
        job.run('killall python || echo nevermind')

    local_batch_size = config.local_batch_size
    base_lr = config.base_lr

    num_workers = num_gpus_per_machine * config.machines
    global_batch_size = local_batch_size * num_workers
    print("using global batch ", global_batch_size)  # 512=8*32*2*1

    # linear LR scaling (https://arxiv.org/abs/1706.02677)
    lr = base_lr * (global_batch_size / BASE_LR_BATCHSIZE)

    # worker parameters with training setup
    root_worker_params = {
        'seed': 1111,
        'adaptive': True,
        'log_interval': 100,
        'eval_interval': 500,
        'max_tokens': int(1.5e9),
        'logdir': job.logdir,
        'lr': lr,
        'batch_size': local_batch_size,
        'eta_min': lr / 10,
        'log_all_workers': args.log_all_workers,
    }

    worker_params = copy.copy(root_worker_params)
    if isinstance(config.worker_params, list) or isinstance(config.worker_params, tuple):
        for cc in config.worker_params:
            worker_params.update(cc)
    else:
        worker_params.update(config.worker_params)

    print("worker params:")
    pprint.pprint(worker_params)

    env_vars = _get_nccl_params()
    if args.skip_wandb:
        env_vars = env_vars + ' WANDB_MODE=dryrun '
    if args.log_all_workers:
        env_vars += ' PYTHONUNBUFFERED=1 '

    for i, task in enumerate(job.tasks):
        dist_params = \
            f'--nproc_per_node={num_gpus_per_machine} ' \
            f'--nnodes={config.machines} --node_rank={i} ' \
            f'--master_addr={job.tasks[0].ip} --master_port={6016}'
        cmd = f'{env_vars} python -m torch.distributed.launch {dist_params} train.py {util.dict_to_args(worker_params)}'
        task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
        task.run(cmd, non_blocking=True)

    print(f"Logging to {job.logdir}")


if __name__ == '__main__':
    main()
