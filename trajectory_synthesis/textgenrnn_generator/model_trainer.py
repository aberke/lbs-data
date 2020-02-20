"""
python model_trainer.py

"""

import os
import textgenrnn


USE_GPU = False
if not USE_GPU:
    # Prevent the environment from seeing the available GPUs (to avoid error on matlaber cluster)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# set higher to train the model for longer
NUM_EPOCHS = 50
# generates sample text from model after given number of epochs
# setting higher than num_epochs to avoid generating samples mid-way
GEN_EPOCHS = NUM_EPOCHS + 1
# maximum number of words to model; the rest will be ignored (word-level model only)
MAX_WORDS = 10000


# --------------------------------------
# Parameters I experiment with tweaking
# The output model files and generated trajectories are named by the
# parameters used in their model training and generation

# consider text both forwards and backward, can give a training boost
RNN_BIDIRECTIONAL = # True, False
# number of tokens to consider before predicting the next
MAX_LENGTH = # 24, 48, 50, 60, 70, 72
# number of LSTM layers
RNN_LAYERS = # 2, 3 
# number of LSTM cells of each layer (128/256 recommended)
RNN_SIZE = # 128, 256
# ignore a random proportion of source tokens each epoch, allowing model to generalize better
DROPOUT = # 0.1, 0.2, 0.3, 0.4
DIM_EMBEDDINGS = # 50, 100, 128

# trained models with the following combinations:
# https://docs.google.com/spreadsheets/d/1XK79VPjp1dqGW6kZUNDHMY_-SiJwKjP3J8OxblNQFcQ/edit?usp=sharing
# --------------------------------------


# Training data is the relabeled trajectories
input_trajectories_filename = '../data/relabeled_trajectories_1_workweek.txt'

name = 'trajectories-rnn_bidirectional:{}-max_len:{}-rnn_layers:{}-rnn_size:{}-dropout:{}-dim_embeddings:{}'.format(
    RNN_BIDIRECTIONAL, MAX_LENGTH, RNN_LAYERS, RNN_SIZE, DROPOUT, DIM_EMBEDDINGS)

print('\ntraining model with %s epochs: %s\n' % (NUM_EPOCHS, name))
textgen = textgenrnn.textgenrnn(name=name)

textgen.train_from_file(
    file_path=input_trajectories_filename,
    new_model=True,
    num_epochs=NUM_EPOCHS,
    gen_epochs=GEN_EPOCHS,
    batch_size=512,
    train_size=1, # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    dropout=DROPOUT,
    validation=False,
    is_csv=False,
    rnn_layers=RNN_LAYERS,
    rnn_size=RNN_SIZE,
    rnn_bidirectional=RNN_BIDIRECTIONAL,
    max_length=MAX_LENGTH,
    dim_embeddings=DIM_EMBEDDINGS,
    word_level=True)

print('done')
