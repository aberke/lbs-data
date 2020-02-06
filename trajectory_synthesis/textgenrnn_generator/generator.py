"""
python generator.py

"""

import os
from datetime import datetime
import json
import textgenrnn


USE_GPU = False
if not USE_GPU:
    # Prevent the environment from seeing the available GPUs (to avoid error on matlaber cluster)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# set higher to train the model for longer
# NUM_EPOCHS = 75
# generates sample text from model after given number of epochs
# setting higher than num_epochs to avoid generating samples mid-way
# GEN_EPOCHS = NUM_EPOCHS + 1
# maximum number of words to model; the rest will be ignored (word-level model only)
# MAX_WORDS = 10000

# parameters I experiment with tweaking...
# number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
# consider text both forwards and backward, can give a training boost
RNN_BIDIRECTIONAL = # True, False
MAX_LENGTH = # 24, 48
RNN_LAYERS = # 2, 3 # number of LSTM layers (>=2 recommended)
RNN_SIZE = # 128, 256   # number of LSTM cells of each layer (128/256 recommended)
DROPOUT = # 0.1, 0.2  # ignore a random proportion of source tokens each epoch, allowing model to generalize better
DIM_EMBEDDINGS = # 50, 100

# trained models with the following combinations:
# https://docs.google.com/spreadsheets/d/1XK79VPjp1dqGW6kZUNDHMY_-SiJwKjP3J8OxblNQFcQ/edit?usp=sharing


# The trained model generates synthetic trajectories where the (home, work) label pairs are
# used as prefixes to produce the remainder of the trajectory.
# In order to better compare the generated data to the real data, we generate 1 trajectory with
# a given (home, work) label pair for each occurance of such a pair in the real (training) data.
# For efficiency, the mapping of (home, work) label pairs to count is precomputed and saved to the
# following file.  The generator model reads the mapping of (home, work) -> count after
# the model is trained in order to generate trajectories with the count for each pair as the number
# of times to use that pair as a prefix.

# Reads in the mapping of (home, work) label pairs -> count
def get_prefixes_to_counts_dict(fname):
    prefixes_to_counts_dict = None
    with open(fname) as json_file:
        prefixes_to_counts_dict = json.load(json_file)
    return prefixes_to_counts_dict


def get_model_generator(model_name):
    return textgenrnn.textgenrnn(weights_path='./{}_weights.hdf5'.format(model_name),
        vocab_path='./{}_vocab.json'.format(model_name),
        config_path='./{}_config.json'.format(model_name),
        name=model_name)

def get_output_filename(model_name, temperature):
    # return './generated-cambridge-{}-temperature:{}.txt'.format(model_name, temperature)
    return './generated-sample-{}-temperature:{}.txt'.format(model_name, temperature)



# Poor hack: Until I properly hack this textgenrnn code, generate vectors of whatever length,
# and then filter them to desired length

seq_length = 122


def filter_to_seq_length(sequences):
    return [seq for seq in sequences if (len(seq.split()) == seq_length)]

def generate_sequences(generator, temperature, prefix, make_num):
    # Current problem: not always getting desired sequence length (TODO: fork and hack on textgenrnn code to fix this)
    # Solution for now: loop to hack around this
    ss = []
    while len(ss) < make_num:
        n = (make_num - len(ss))*2
        generated_sequences = textgenrnn.utils.synthesize(
            [generator], n=n, prefix=prefix, temperature=[temperature],
            return_as_list=True, max_gen_length=seq_length+1, stop_tokens=['hack'])
        ss += filter_to_seq_length(generated_sequences)
    return ss[:make_num]


# Generate the sequences!

# Generate multiplier synthetic sequences for every real sequence
count_multiplier = 1

# The trained model has a name
model_name = 'trajectories-rnn_bidirectional:{}-max_len:{}-rnn_layers:{}-rnn_size:{}-dropout:{}-dim_embeddings:{}'.format(
    RNN_BIDIRECTIONAL, MAX_LENGTH, RNN_LAYERS, RNN_SIZE, DROPOUT, DIM_EMBEDDINGS)

generate_temperatures = [0.8, 0.9, 1.0]

print('\nwill generate trajectories for temperatures %s and output to files %s\n' % (generate_temperatures, [get_output_filename(model_name, t) for t in generate_temperatures]))


# NOTE: Some generation wass done for data Cambridge specific data, only using prefixes where the home label is a Cambridge GEOID
input_trajectories_prefixes_to_counts_filename = '../data/relabeled_trajectories_1_workweek_prefixes_to_counts_sample_2000.json'
# input_trajectories_prefixes_to_counts_filename = '../data/relabeled_cambridge_trajectories_1_workweek_prefixes_to_counts.json'
prefixes_to_counts_dict = get_prefixes_to_counts_dict(input_trajectories_prefixes_to_counts_filename)

# generate with a variety of temperatures
generator = get_model_generator(model_name)
print('\nwill generate trajectories for temperatures %s and output to files %s\n' % (generate_temperatures, [get_output_filename(model_name, t) for t in generate_temperatures]))
for temperature in generate_temperatures:
    output_fname = get_output_filename(model_name, temperature)
    print('%s : generating trajectories and saving to file: %s' % (datetime.now(), output_fname))
    sequences = []
    i = 0
    for prefix_labels, count in prefixes_to_counts_dict.items():
        if i % 100 == 0:
            print('%s : %s : generated %s sequences...' % (datetime.now(), i, len(sequences)))
        i += 1

        make_num = count*count_multiplier
        # Add an extra space so that the work prefix label has proper end and model continues to next label
        prefix = '%s ' % prefix_labels
        sequences += generate_sequences(generator, temperature, prefix, make_num=make_num)
    print('writing sequences to file', output_fname)
    with open(output_fname, 'w') as f:
        for seq in sequences:
            f.write('{}\n'.format(seq))

print('done')
