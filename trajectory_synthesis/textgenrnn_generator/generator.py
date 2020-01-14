"""
python generator.py

"""

import os
import json
import textgenrnn


USE_GPU = False
if not USE_GPU:
    # Prevent the environment from seeing the available GPUs (to avoid error on matlaber cluster)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"


# set higher to train the model for longer
NUM_EPOCHS = 1 #50
# generates sample text from model after given number of epochs
# setting higher than num_epochs to avoid generating samples mid-way
GEN_EPOCHS = NUM_EPOCHS + 1
# consider text both forwards and backward, can give a training boost
RNN_BIDIRECTIONAL = True
# maximum number of words to model; the rest will be ignored (word-level model only)
MAX_WORDS = 10000

# parameters I experiment with tweaking...
# number of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
MAX_LENGTH = 50
RNN_LAYERS = 3  # number of LSTM layers (>=2 recommended)
RNN_SIZE = 128   # number of LSTM cells of each layer (128/256 recommended)
DROPOUT = 0.1  # ignore a random proportion of source tokens each epoch, allowing model to generalize better
DIM_EMBEDDINGS = 128



# Training data is the relabeled trajectories
input_trajectories_filename = '../data/relabeled_trajectories_1_workweek.txt'

# The trained model generates synthetic trajectories where the (home, work) label pairs are
# used as prefixes to produce the remainder of the trajectory.
# In order to better compare the generated data to the real data, we generate 1 trajectory with
# a given (home, work) label pair for each occurance of such a pair in the real (training) data.
# For efficiency, the mapping of (home, work) label pairs to count is precomputed and saved to the
# following file.  The generator model reads the mapping of (home, work) -> count after
# the model is trained in order to generate trajectories with the count for each pair as the number
# of times to use that pair as a prefix.
input_trajectories_prefixes_to_counts_filename = '../data/relabeled_trajectories_1_workweek_prefixes_to_counts.json'

generate_temperatures = [0.8, 0.9, 1.0]

name = 'trajectories-max_len:{}-rnn_layers:{}-rnn_size:{}-dropout:{}-dim_embeddings:{}'.format(
    MAX_LENGTH, RNN_LAYERS, RNN_SIZE, DROPOUT, DIM_EMBEDDINGS)

print('\ntraining model with %s epochs: %s\n' % (NUM_EPOCHS, name))

def get_output_filename(temperature):
    return './output/generated-{}-temperature:{}.txt'.format(name, temperature)

print('\nwill generate trajectories for temperatures %s and output to files %s\n' % (generate_temperatures, [get_output_filename(t) for t in generate_temperatures]))


textgen = textgenrnn.textgenrnn(name=name)

textgen.train_from_file(
    file_path=input_trajectories_filename,
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


def filter_to_seq_length(sequences):
    return [seq for seq in sequences if (len(seq.split()) == seq_length)]

def generate_sequences(temperature, prefix, make_num=100):
    # Current problem: not always getting desired sequence length (TODO: fork and hack on textgenrnn code to fix this)
    # Solution for now: loop to hack around this
    ss = []
    while len(ss) < make_num:
        n = (make_num - len(ss))*2
        generated_sequences = textgenrnn.utils.synthesize(
            [generator], n, prefix=prefix, temperature=[temperature],
            return_as_list=True, max_gen_length=seq_length+1, stop_tokens=['hack'])
        ss += filter_to_seq_length(generated_sequences)
    return ss[:make_num]


# Generate the sequences!

# Read in the mapping of (home, work) label pairs -> count
def get_prefixes_to_counts_dict():
    prefixes_to_counts_dict = None
    with open(input_trajectories_prefixes_to_counts_filename) as json_file:
        prefixes_to_counts_dict = json.load(json_file)

prefixes_to_counts_dict = get_prefixes_to_counts_dict()

# generate with a variety of temperatures
for temperature in generate_temperatures:
    output_fname = get_output_filename(temperature)
    print('generating trajectories and saving to file: %s' % output_fname)
    sequences = []
    for prefix_labels, count in prefixes_to_counts_dict.items():
        # Add an extra space so that the work prefix label has proper end and model continues to next label
        prefix = '%s ' % prefix_labels
        sequences += generate_sequences(temperature, prefix, make_num=count)
    with open(output_fname, 'w') as f:
        for seq in sequences:
            f.write('{}\n'.format(seq))


print('done')
