# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-wiki'
eval_interval = 20 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'wiki'
wandb_run_name = 'mini-gpt-wiki'

dataset = 'wiki'
gradient_accumulation_steps = 1
batch_size = 8 # batch size
block_size = 512 # context of up to 4096 previous tokens

# baby GPT model :)
n_layer = 3
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 1_000_000_000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 10 # not super necessary potentially
device = 'mps'

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

