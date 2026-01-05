# Learning to play Tic-Tac-Toe in Jax

* https://joe-antognini.github.io/ml/jax-tic-tac-toe

This is a minimal setup to train a neural network to play Tic-Tac-Toe using
reinforcement learning.  The model learns perfect play in about 15 seconds or
so.

To set it up and run the script, run the following:

```sh
git clone https://github.com/charbull/jax-tic-tac-toe.git
cd jax-tic-tac-toe
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python tic_tac_toe.py
```

This should produce output like:

```
Platform 'METAL' is experimental and not all JAX functionality may be correctly supported!
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1767626936.045141 60009208 mps_client.cc:510] WARNING: JAX Apple GPU support is experimental and not all JAX functionality is correctly supported!
Metal device set to: Apple M4 Pro

systemMemory: 48.00 GB
maxCacheSize: 18.00 GB

I0000 00:00:1767626936.055691 60009208 service.cc:145] XLA service 0x6000017ccf00 initialized for platform METAL (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1767626936.055700 60009208 service.cc:153]   StreamExecutor device (0): Metal, <undefined>
I0000 00:00:1767626936.056609 60009208 mps_client.cc:406] Using Simple allocator.
I0000 00:00:1767626936.056615 60009208 mps_client.cc:384] XLA backend will use up to 38654230528 bytes on device 0 for SimpleAllocator.
Step 0: Wins: 42.97%  Ties: 11.23%  Losses: 45.80%
Step 207; Wins: 92.09%  Ties: 4.20%  Losses: 3.71%                                                                                                                        Step 405; Wins: 90.14%  Ties: 7.62%  Losses: 2.25%                                                                                                                        Step 603; Wins: 91.11%  Ties: 8.40%  Losses: 0.49%                                                                                                                        Step 801; Wins: 94.53%  Ties: 5.08%  Losses: 0.39%                                                                                                                        Step 1008; Wins: 95.41%  Ties: 4.59%  Losses: 0.00%                                                                                                                        Step 1206; Wins: 95.31%  Ties: 4.59%  Losses: 0.10%                                                                                                                        Step 1404; Wins: 95.12%  Ties: 4.88%  Losses: 0.00%                                                                                                                        Step 1602; Wins: 95.02%  Ties: 4.98%  Losses: 0.00%                                                                                                                        Step 1800; Wins: 94.04%  Ties: 5.86%  Losses: 0.10%                                                                                                                        Step 2007; Wins: 92.68%  Ties: 7.32%  Losses: 0.00%                                                                                                                        Step 2205; Wins: 92.97%  Ties: 7.03%  Losses: 0.00%                                                                                                                        Step 2403; Wins: 93.46%  Ties: 6.54%  Losses: 0.00%                                                                                                                        2502it [00:18, 132.04it/s]                                                                                                                         Step 2502: Wins: 93.65%  Ties: 6.35%  Losses: 0.00%
```

If you play this model against itself, it will produce a game with perfect play
like this:

![Tic-Tac-Toe with perfect play](perfect_play.svg)

## MacOS
```sh
python -m venv venv
source venv/bin/activate
pip install -r mac_requirements.txt
export ENABLE_PJRT_COMPATIBILITY=1
python tic_tac_toe.py
```
