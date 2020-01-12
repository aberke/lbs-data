# Trajectory Generation with textgenrnn


Project: https://github.com/minimaxir/textgenrnn


```
# Install the dependency
pip install --user textgenrnn

# Run the script
python3 generator.py
```

For running as a job on a remote machine, we do:

```
$ nohup python3 generator.py > process.out 2> process.err < /dev/null &
```

And then later take a look at what happend in `./output`


Can also work in colab: https://colab.research.google.com/drive/1PXV7UdzXPEvazpd_Lwk1Q85YjtIFAHUk

