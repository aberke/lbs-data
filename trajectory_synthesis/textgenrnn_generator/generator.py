"""
python generator.py

"""

import os
from datetime import datetime
import textgenrnn


USE_GPU = False
if not USE_GPU:
    # Prevent the environment from seeing the available GPUs (to avoid error on matlaber cluster)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"


NUM_EPOCHS = 50  # set higher to train the model for longer
GEN_EPOCHS = 25  # generates sample text from model after given number of epochs
RNN_LAYERS = 3  # number of LSTM layers (>=2 recommended)
RNN_SIZE = 128   # number of LSTM cells of each layer (128/256 recommended)
DROPOUT = 0.1  # ignore a random proportion of source tokens each epoch, allowing model to generalize better
RNN_BIDIRECTIONAL = True  # consider text both forwards and backward, can give a training boost
DIM_EMBEDDINGS = 128

MAX_LENGTH = 50   # number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
MAX_WORDS = 10000  # maximum number of words to model; the rest will be ignored (word-level model only)


input_filename = '../data/relabeled_trajectories_1_workweek.txt'

textgen = textgenrnn.textgenrnn(name='synthetic_trajectories')

textgen.train_from_file(
    file_path=input_filename,
    new_model=True,
    num_epochs=NUM_EPOCHS,
    gen_epochs=GEN_EPOCHS,
    batch_size=1024,
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


# Poor hack: Until I properly hack this textgenrnn code, generate vectors of whatever length,
# and then filter them to desired length

seq_length = 122
N = 1000  # make this many

def filter_to_seq_length(sequences):
    return [seq for seq in sequences if (len(seq.split()) == seq_length)]

def generate_sequences(temperature, n, prefix=None):
    ss = textgenrnn.utils.synthesize([textgen], n, prefix=prefix, temperature=[temperature],
                                 return_as_list=True, max_gen_length=seq_length+1, stop_tokens=['hack'])
    return filter_to_seq_length(ss)


def get_output_filename(temperature, prefix=None):
    return './output/generated-epochs:{}:{}-temperature:{}-prefix:{}-{}.txt'.format(
        NUM_EPOCHS, GEN_EPOCHS, temperature, prefix, datetime.now().strftime('%Y%m%d'))


# generate with a variety of temperatures
for temp in [0.8, 0.9, 1.0]:
    output_fname = get_output_filename(temp)
    print('using input file', input_filename, 'generating to ', output_fname)
    sequences = generate_sequences(temp, N)
    with open(output_fname, 'w') as f:
        for seq in sequences:
            f.write('{}\n'.format(seq))


print('done')
