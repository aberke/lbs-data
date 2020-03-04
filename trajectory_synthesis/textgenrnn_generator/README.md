# Trajectory Generation with textgenrnn


Project: https://github.com/minimaxir/textgenrnn


```
# Install the dependency
pip install --user textgenrnn

# Run the scripts

python3 model_trainer.py

... later after model training completes...

python3 generator.py
```

For running as a job on a remote machine, we do:

```
$ nohup python3 model_trainer.py > train_process.out 2> train_process.err < /dev/null &

... later after model training completes...
$ nohup python3 generator.py > gen_process.out 2> gen_process.err < /dev/null &
```

And then later take a look at what happend in `./output`


Can also work in colab: https://colab.research.google.com/drive/1PXV7UdzXPEvazpd_Lwk1Q85YjtIFAHUk

