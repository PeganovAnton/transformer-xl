# Scalable training of Transformer-XL

Extending [Transformer-XL](https://github.com/kimiyoung/transformer-xl) for distributed training. (medium [post](https://medium.com/south-park-commons/scaling-transformer-xl-to-128-gpus-85849508ec35))

<img src="https://raw.githubusercontent.com/cybertronai/transformer-xl/master/scaling.png">


## One machine
Reproduce 24.617 test perplexity on wikitext-103 after 2 hours 54 minutes and one machine with 180M Transformer-XL (internal logs: east/txl.03).
```
aws configure
pip install -r requirements.txt
python launch.py --config=one_machine
```

Command above will reserve p3dn instance, setup dependencies and start training. You can see what it's doing by doing one of:
1. Connect to instance using ssh and attach to tmux session. (`ncluster mosh` does this automatically)
2. Call `launch_tensorboard.py` to spin up TensorBoard instance and look at the latest graphs that appeared.


## Several machines
Reproduce 90 seconds per epoch on 180M Transformer-XL network and 8-machines.

```
python launch.py --config=eight_machines
```


## Throughput tests
Training throughput numbers for 277M parameter Transformer-XL
```
python launch.py --config=test_1    # 446  examples/sec
python launch.py --config=test_2    # 860  examples/sec
python launch.py --config=test_4    # 1736 examples/sec
python launch.py --config=test_8    # 3031 examples/sec
python launch.py --config=test_16   # 5904 examples/sec
```

## Locally on single GPU

Follow instructions on original Transformer-XL repo to [get data](https://github.com/kimiyoung/transformer-xl/tree/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch#data-prepration) and put it under `data/wikitext-103`, then

```
bash single_machine.sh
```


## Notes

- These experiments automatically launch at spot instance prices. If you prefer to launch instances as on-demand , add `--nospot` flag to your launch.py command.

- By default, instances stop automatically 10 minutes after completion or 60 minutes if error is detected. You could stop instances manually using `ncluster stop` command or terminate them with `ncluster kill` commands.

- Use ncluster [command-line tool](https://github.com/yaroslavvb/ncluster#command-line-tool) to interact with your instances
