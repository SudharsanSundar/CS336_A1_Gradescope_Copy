import pprint
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import torch
import torch.nn as nn
import numpy as np
import random
import wandb
from cs336_basics import transformer, training, tokenizer_class_2, tokenizer_fast
import json
import argparse
from tqdm import tqdm
import time

run = wandb.init(
    project='train-transformer'
)
ppr = pprint.PrettyPrinter()


def training_loop(training_args: dict):
    # Digest training args
    start_time = time.time()
    wandb.config = training_args

    train_data_fp = training_args['train_data_fp']
    val_data_fp = training_args['val_data_fp']
    d_model = training_args['d_model']
    num_heads = training_args['num_heads']
    d_ff = training_args['d_ff']
    attn_pdrop = training_args['attn_pdrop']
    residual_pdrop = training_args['residual_pdrop']
    vocab_size = training_args['vocab_size']
    context_length = training_args['context_length']
    num_layers = training_args['num_layers']
    lr = training_args['lr']
    betas = training_args['betas']
    eps = training_args['eps']
    weight_decay = training_args['weight_decay']
    load_from_checkpoint = training_args['load_from_checkpoint']
    num_steps = training_args['num_steps']
    batch_size = training_args['batch_size']
    epochs = training_args['epochs']
    device = training_args['device']
    max_norm = training_args['max_norm']
    checkpoint_interval = training_args['checkpoint_interval']
    checkpoint_fp = training_args['checkpoint_fp']
    validation_interval = training_args['validation_interval']
    num_val_examples = training_args['num_val_examples']
    max_lr = training_args['max_lr']
    min_lr = training_args['min_lr']
    lr_warmup = training_args['lr_warmup']
    lr_cooldown = training_args['lr_cooldown']
    use_lr_scheduler = training_args['use_lr_scheduler']
    tokenizer_vocab_fp = training_args['tokenizer_vocab_fp']
    tokenizer_merges_fp = training_args['tokenizer_merges_fp']
    special_tokens = training_args['special_tokens']

    # TODO add args for evaluation

    # Load tokenized data via memmap
    train_data = np.memmap(train_data_fp, dtype=np.uint16, mode='r')
    validation_data = np.memmap(filename=val_data_fp, dtype=np.uint16, mode='r')
    train_data = train_data.astype('int')
    validation_data = validation_data.astype('int')

    # Create model
    model = transformer.Transformer(d_model=d_model,
                                    num_heads=num_heads,
                                    d_ff=d_ff,
                                    attn_pdrop=attn_pdrop,
                                    residual_pdrop=residual_pdrop,
                                    vocab_size=vocab_size,
                                    context_length=context_length,
                                    num_layers=num_layers,
                                    weights=None,
                                    tokenizer_vocab_fp=tokenizer_vocab_fp,
                                    tokenizer_merges_fp=tokenizer_merges_fp,
                                    special_tokens=special_tokens)

    # Create optimizer setup
    loss_fn = transformer.CrossEntropyLoss()

    optimizer = training.AdamW(params=model.parameters(),
                               lr=lr,
                               betas=betas,
                               eps=eps,
                               weight_decay=weight_decay,
                               device=device)

    lr_scheduler = training.lr_cosine_schedule

    # Load state from checkpoint if desired
    if load_from_checkpoint is not None:
        print('Loading from checkpoint...')
        starting_training_its = training.load_checkpoint(src=load_from_checkpoint,
                                                         model=model,
                                                         optimizer=optimizer)
        print(f'Starting back up from iteration {starting_training_its}...')
        optimizer.to('cuda')
    else:
        starting_training_its = 0

    # Determine total training iterations
    total_training_its = min(num_steps, len(train_data) / (batch_size + context_length))

    # Set model to train
    model.to(torch.device(device))
    model.train()

    # Run training loop
    train_losses = []
    val_losses = []
    for i in range(epochs):
        for j in tqdm(range(starting_training_its, total_training_its, 1), total=total_training_its, initial=starting_training_its):
            # Update lr
            if use_lr_scheduler:
                for group in optimizer.param_groups:
                    group['lr'] = lr_scheduler(t=j,
                                               lr_max=max_lr,
                                               lr_min=min_lr,
                                               t_warmup=lr_warmup,
                                               t_cooldown=lr_cooldown)

            # Get batch of data
            # input_batch, target_batch = training.load_batch(arr=train_data[j * context_length * batch_size:(j + 1) * context_length * batch_size],
            #                                                 batch_size=batch_size,
            #                                                 context_length=context_length,
            #                                                 device=device)
            input_batch, target_batch = training.load_batch(
                arr=train_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device)

            # Get model preds
            pred_batch = model(in_indices=input_batch)

            # Get loss
            loss = loss_fn(pred_batch, target_batch)
            train_losses.append(loss.item())
            wandb.log({'train-loss': loss.item()})

            # Backprop loss
            loss.backward()

            # Clip gradients
            training.clip_gradients(model.parameters(), max_norm=max_norm)

            # Step with optimizer
            optimizer.step()
            optimizer.zero_grad()

            # Save checkpoint every n training steps
            if j % checkpoint_interval == 0:
                training.save_checkpoint(model=model, optimizer=optimizer, iteration=i, out=checkpoint_fp)
                print('-'*50, '\n~~SAVED CHECKPOINT~~\n', '-'*50)

            # Evaluate on validation set every n training steps
            if j % validation_interval == 0:
                val_loss = run_validation(model=model,
                                          validation_data=validation_data,
                                          context_length=context_length,
                                          device=device,
                                          loss_fn=loss_fn,
                                          num_examples=num_val_examples)
                val_losses.append(val_loss)
                wandb.log({'val-loss': val_loss})
                wandb.log({'wallclock-time-elapsed': time.time() - start_time})
                model.train()
                print('~~DONE VAL~~\n', '-'*50)

    training.save_checkpoint(model=model, optimizer=optimizer, iteration=total_training_its, out=checkpoint_fp)
    print('-' * 50, '\n~~SAVED CHECKPOINT~~\n', '-' * 50)

    end_time = time.time()
    wandb.log({'total-training-time-seconds': end_time - start_time})

    return train_losses, val_losses


def run_validation(model, validation_data, context_length, device, loss_fn, num_examples=100, num_rounds=4):
    model.eval()
    print('-'*50, '\nBEGINNING VAL')

    total_loss = 0
    for i in range(num_rounds):
        # Get batch of data
        # randstart = random.randint(0, len(validation_data) - 1 - context_length * num_examples)
        # input_batch, target_batch = training.load_batch(arr=validation_data[randstart:randstart + context_length * num_examples],
        #                                                 batch_size=num_examples,
        #                                                 context_length=context_length,
        #                                                 device=device)
        input_batch, target_batch = training.load_batch(
            arr=validation_data,
            batch_size=num_examples,
            context_length=context_length,
            device=device)

        # Get model preds
        pred_batch = model(in_indices=input_batch)

        # Get loss
        loss = loss_fn(pred_batch, target_batch)
        total_loss += loss.item()

    # Display results
    print(f'\n{num_examples} random ex seqs for {num_rounds} rounds, {num_examples * num_rounds} total.\n\n', '   VALIDATION LOSS:', total_loss / num_rounds)

    return total_loss / num_rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_args_fp', type=str)
    training_args_fp = parser.parse_args().train_args_fp
    training_args = json.load(open(training_args_fp))
    print('TRAINING ARGS:')
    ppr.pprint(training_args)

    print('-'*50, '\nDone loading training args. Starting training...')

    train_loss, val_loss = training_loop(training_args)
    print('\n', '-'*50, '\nDONE TRAINING')


if __name__ == '__main__':
    main()
