# Learning to play Tic-Tac-Toe in Jax

* https://joe-antognini.github.io/ml/jax-tic-tac-toe

This is a minimal setup to train a neural network to play Tic-Tac-Toe using
reinforcement learning.  The model learns perfect play in about 15 seconds or
so.

To set it up and run the script, run the following:

```sh
git clone https://github.com/joe-antognini/jax-tic-tac-toe.git
cd jax-tic-tac-toe
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python tic_tac_toe.py
```

This should produce output like:

```
Step 0: Wins: 46.39%  Ties: 17.77%  Losses: 35.84%
Step 207; Wins: 92.19%  Ties: 4.59%  Losses: 3.22%
Step 405; Wins: 91.41%  Ties: 7.23%  Losses: 1.37%
Step 603; Wins: 90.23%  Ties: 9.38%  Losses: 0.39%
Step 801; Wins: 93.16%  Ties: 6.25%  Losses: 0.59%
Step 1008; Wins: 93.65%  Ties: 6.05%  Losses: 0.29%
Step 1206; Wins: 92.87%  Ties: 7.13%  Losses: 0.00%
Step 1404; Wins: 93.16%  Ties: 6.84%  Losses: 0.00%
Step 1602; Wins: 93.26%  Ties: 6.54%  Losses: 0.20%
Step 1800; Wins: 93.07%  Ties: 6.74%  Losses: 0.20%
Step 2007; Wins: 93.95%  Ties: 6.05%  Losses: 0.00%
Step 2205; Wins: 94.34%  Ties: 5.66%  Losses: 0.00%
Step 2403; Wins: 94.34%  Ties: 5.66%  Losses: 0.00%
2502it [00:15, 161.33it/s]
Step 2502: Wins: 94.14%  Ties: 5.86%  Losses: 0.00%
```

If you play this model against itself, it will produce a game with perfect play
like this:

![Tic-Tac-Toe with perfect play](perfect_play.svg)
