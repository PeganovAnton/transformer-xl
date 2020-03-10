import math
import os
import time
from typing import Tuple

import torch
import wandb
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange, tqdm
from transformers import PreTrainedModel, AdamW, get_cosine_schedule_with_warmup

from hf_training.eval import evaluate
from hf_training.log import timeit, logger
from hf_training.utils import set_seed, _rotate_checkpoints, save_checkpoint


def train(args, train_data_iterator, eval_data_iterator, model: PreTrainedModel, best_eval_loss: float) -> Tuple[int, float]:
    """ Train the model """
    total_time_start = time.time()

    args.examples_per_step = args.train_batch_size * args.gradient_accumulation_steps
    args.tokens_per_step = args.examples_per_step * args.block_size

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_data_iterator) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_data_iterator) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate / 32 * args.examples_per_step, eps=args.adam_epsilon
    )
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_tokens / args.tokens_per_step), num_training_steps=t_total
    )
    logger.info(f"Warmup for {int(args.warmup_tokens / args.tokens_per_step)} steps")

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    if args.local_rank in [-1, 0]:
        wandb.init(project="GPT-2 git", name="sample_name", config=args)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data_iterator))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_data_iterator) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                len(train_data_iterator) // args.gradient_accumulation_steps
            )

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    unwrapped_model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    if args.model_type == "gpt-2":
        wandb.watch(unwrapped_model, log_freq=args.logging_steps)

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args.seed, args.n_gpu > 0)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_data_iterator, desc="Iteration", disable=args.local_rank not in [-1, 0])

        tr_loss, logging_loss = 0.0, 0.0
        training_time = time.time()
        steps_past, consumed_tokens = 0, 0
        mems = tuple()

        for step, batch in enumerate(epoch_iterator):
            if args.model_type == "gpt-2":
                inputs, labels = (batch, batch)
            else:
                assert args.model_type == "txl"
                inputs, labels = batch
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            assert inputs.ndimension() == 2 and inputs.size() == (args.train_batch_size, args.block_size)
            assert labels.ndimension() == 2 and labels.size() == (args.train_batch_size, args.block_size)
            consumed_tokens += args.train_batch_size * inputs.size(1)
            steps_past += 1

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            if args.model_type == "gpt-2":
                loss, *_ = model(inputs, labels=labels)
            else:
                assert args.model_type == "txl"
                loss, _, *mems = model(*mems, input_ids=inputs, labels=labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            with timeit(tag="backward_time", step=global_step, noop=args.local_rank not in [-1, 0]):
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()

                if args.local_rank in [-1, 0]:
                    # Log train metrics
                    wandb.log(
                        {
                            "train/lr": scheduler.get_lr()[0],
                            "train/loss": (tr_loss - logging_loss) / steps_past,
                            "train/perplexity": math.exp((tr_loss - logging_loss) / steps_past),
                        },
                        step=global_step,
                    )
                    logging_loss = tr_loss

                    # Log times
                    total_time = time.time() - total_time_start
                    since_last_eval_time = time.time() - training_time
                    step_time = since_last_eval_time / steps_past
                    tokens_per_sec = 1 / (since_last_eval_time / consumed_tokens)

                    wandb.log(
                        {
                            "times/step_time": step_time,
                            "times/tokens_per_sec": tokens_per_sec,
                            "times/money": total_time / 3600 * 24.5,
                        },
                        step=global_step,
                    )

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log eval metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, eval_data_iterator)
                        for key, value in results.items():
                            wandb.log({"eval/{}".format(key): value}, step=global_step)
                        eval_loss = results["loss"]
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            save_checkpoint(args, 0, model, optimizer, scheduler, checkpoint_prefix="checkpoint-best")

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(args, global_step, model, optimizer, scheduler, checkpoint_prefix="checkpoint")

                global_step += 1

                training_time = time.time()
                steps_past, consumed_tokens = 0, 0

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step
