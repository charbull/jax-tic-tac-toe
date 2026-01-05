import dataclasses
from typing import NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
import optax
import pgx
from flax import nnx
from pgx.tic_tac_toe import TicTacToe
from tqdm.auto import tqdm

BOARD_SIZE = 9


class Transition(NamedTuple):
    """A single transition to train on."""

    state: pgx.State
    action: jax.Array
    next_state: pgx.State


@dataclasses.dataclass(frozen=True)
class HParams:
    """Hyperparameters to train the neural network."""

    batch_size: int = 2048
    eps_start: float = 0.9
    eps_end: float = 0.05
    learning_rate: float = 2e-3
    n_neurons: int = 128
    n_train_steps: int = 2500
    tau: float = 0.005
    warm_up_steps: int = 128
jax.tree_util.register_pytree_node(HParams, lambda x: ((), x), lambda x, _: x)


@dataclasses.dataclass
class TrainState:
    """State associated with model training."""

    policy_net: nnx.Module
    target_net: nnx.Module
    optimizer: nnx.Optimizer
    rng_key: jax.Array
    cur_step: int = 0


@dataclasses.dataclass
class GameStatistics:
    """Tracks wins, losses, and ties."""

    n_wins: int
    n_ties: int
    n_losses: int

    @property
    def games_played(self):
        return self.n_wins + self.n_ties + self.n_losses

    @property
    def win_frac(self):
        return self.n_wins / self.games_played

    @property
    def loss_frac(self):
        return self.n_losses / self.games_played

    @property
    def tie_frac(self):
        return self.n_ties / self.games_played

    def __repr__(self):
        return (
            f'Wins: {100 * self.win_frac:.2f}%  '
            f'Ties: {100 * self.tie_frac:.2f}%  '
            f'Losses: {100 * self.loss_frac:.2f}%'
        )


class DQN(nnx.Module):
    """A simple fully connected Q-network for Tic-Tac-Toe."""

    def __init__(self, *, rngs: nnx.Rngs, hparams: HParams):
        self.hparams = hparams
        self.linear1 = nnx.Linear(BOARD_SIZE, hparams.n_neurons, rngs=rngs)
        self.linear2 = nnx.Linear(
            hparams.n_neurons, hparams.n_neurons, rngs=rngs
        )
        self.linear3 = nnx.Linear(hparams.n_neurons, BOARD_SIZE, rngs=rngs)

    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = x[..., 0] - x[..., 1]
        x = jnp.reshape(x, (-1, BOARD_SIZE))
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        return nnx.tanh(self.linear3(x))


@nnx.jit
def act_randomly(rng, state):
    """Select a single valid action from a uniform distribution."""
    probs = state.legal_action_mask / state.legal_action_mask.sum()
    logits = jnp.maximum(jnp.log(probs), jnp.finfo(probs.dtype).min)
    return jax.random.categorical(rng, logits, axis=-1)


def sample_action_eps_greedy(
    rng, state: pgx.State, policy_net: DQN, eps: float, batch_size: int
) -> jax.Array:
    """Select actions using an epsilon-greed strategy.

    Given a number epsilon, actions will be chosen from a uniform distribution
    with probability epsilon, and otherwise the best action will be selected
    according to `policy_net`.  Note that this sampling is done in a
    per-example basis, so within a batch some actions may be selected randomly,
    and others will be selected using `policy_net`.

    """
    rng, subkey = jax.random.split(rng)
    eps_sample = jax.random.uniform(subkey, [batch_size])
    best_actions = select_best_action(state, policy_net)
    random_actions = act_randomly(rng, state)

    eps_mask = eps_sample > eps
    return best_actions * eps_mask + random_actions * (1 - eps_mask)


@nnx.jit
def select_best_action(state: pgx.State, policy_net: DQN):
    """Choose the best action according to `policy_net`."""
    logits = policy_net(state.observation)
    return jnp.argmax(
        logits * state.legal_action_mask
        + jnp.finfo(logits.dtype).min * ~state.legal_action_mask,
        axis=-1,
    )


def select_action(game_state, train_state, hparams) -> jax.Array:
    """Choose an action with epsilon-greedy sampling and decay epsilon.

    This will apply a linear decay to epsilon over the course of training and
    then use epsilon-greedy sampling to choose an action.

    """
    eps = (
        (hparams.eps_start - hparams.eps_end)
        * (1 - train_state.cur_step / hparams.n_train_steps)
        + hparams.eps_end
    )
    train_state.rng_key, subkey = jax.random.split(train_state.rng_key)
    return sample_action_eps_greedy(
        subkey, game_state, train_state.policy_net, eps, hparams.batch_size
    )


def loss_fn(policy_net, next_state_values, state, action):
    """Apply a Huber loss to the policy net Q-values and next state values."""
    batch_size = state.observation.shape[0]
    state_action_values = policy_net(
        state.observation
    )[jnp.arange(batch_size), action]
    loss = optax.huber_loss(state_action_values, next_state_values)
    mask = (~state.terminated).astype(jnp.float32)
    return (loss * mask).mean()


@partial(nnx.jit, static_argnames='hparams')
def train_step(
    policy_net, target_net, optimizer, transition, hparams
):
    """Take a single training step.

    This uses the Bellman equations to assess the value of the current state as
    the reward from the next state plus the value of the next state (according
    to the target net).  Then a Huber loss is applied between this value and
    the value predicted by the policy net on the current state.

    """
    state, action, next_state = transition

    next_state_rewards = next_state.rewards[
        jnp.arange(hparams.batch_size), next_state.current_player
    ]

    best_next_state = jnp.max(
        target_net(next_state.observation) * next_state.legal_action_mask
        - ~next_state.legal_action_mask,
        axis=1,
    )
    termination_mask = (~next_state.terminated).astype(jnp.float32)

    # Flip the sign since it's the other player's turn
    next_state_values = -(
        next_state_rewards + termination_mask * best_next_state
    )

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(policy_net, next_state_values, state, action)
    optimizer.update(grads)

    _, policy_params = nnx.split(policy_net)
    _, target_params = nnx.split(target_net)
    target_params = jax.tree.map(
        lambda p, t: (1 - hparams.tau) * t + hparams.tau * p,
        policy_params,
        target_params,
    )
    nnx.update(target_net, target_params)


def run_game(init_fn, step_fn, train_state: TrainState, hparams: HParams):
    """Iterate over a single batch of games, training on every step."""
    train_state.rng_key, subkey = jax.random.split(train_state.rng_key)
    keys = jax.random.split(subkey, hparams.batch_size)
    state = init_fn(keys)

    while not (state.terminated | state.truncated).all():
        train_state.rng_key, subkey = jax.random.split(train_state.rng_key)
        action = select_action(state, train_state, hparams)
        next_state = step_fn(state, action)
        transition = Transition(
            state=state, action=action, next_state=next_state
        )

        train_step(
            train_state.policy_net,
            train_state.target_net,
            train_state.optimizer,
            transition,
            hparams,
        )

        state = next_state
        train_state.cur_step += 1


def measure_game_stats_against_random_player(
    key, init_fn, step_fn, policy_net, n_games: int = 1024
) -> GameStatistics:
    n_wins = 0
    n_losses = 0

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_games)
    state = init_fn(keys)

    while not (state.terminated | state.truncated).all():
        key, subkey = jax.random.split(key)
        random_actions = act_randomly(subkey, state)
        best_actions = select_best_action(state, policy_net)

        # Policy net is player 0, random player is player 1.
        actions = (
            random_actions * state.current_player
            + best_actions * (1 - state.current_player)
        )

        state = step_fn(state, actions)

        n_wins += jnp.sum(state.rewards[:, 0] == 1)
        n_losses += jnp.sum(state.rewards[:, 0] == -1)

    n_ties = n_games - n_wins - n_losses
    return GameStatistics(
        n_wins=n_wins,
        n_ties=n_ties,
        n_losses=n_losses,
    )


def train_model(seed: int = 1, eval_steps: int = 200):
    """Train a neural network to play Tic-Tac-Toe."""
    env = TicTacToe()
    init_fn = jax.vmap(env.init)
    step_fn = nnx.jit(jax.vmap(env.step))

    hparams = HParams()
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    policy_net = DQN(hparams=hparams, rngs=nnx.Rngs(subkey))
    target_net = DQN(hparams=hparams, rngs=nnx.Rngs(subkey))

    lr_schedule = optax.schedules.linear_schedule(
        init_value=hparams.learning_rate,
        end_value=0,
        transition_steps=hparams.n_train_steps,
    )
    optimizer = nnx.Optimizer(
        policy_net, optax.adamw(lr_schedule), wrt=nnx.Param
    )

    train_state = TrainState(
        policy_net=policy_net,
        target_net=target_net,
        optimizer=optimizer,
        rng_key=key,
    )

    stats = measure_game_stats_against_random_player(
        key, init_fn, step_fn, train_state.policy_net
    )
    all_game_stats = [stats]
    print(f'Step 0: {stats}')

    prev_step = 0
    with tqdm(total=hparams.n_train_steps) as pbar:
        while train_state.cur_step < hparams.n_train_steps:
            run_game(init_fn, step_fn, train_state, hparams)
            if train_state.cur_step // eval_steps != prev_step // eval_steps:
                stats = measure_game_stats_against_random_player(
                    key, init_fn, step_fn, train_state.policy_net
                )
                all_game_stats.append(stats)
                pbar.write(f'Step {train_state.cur_step}; {stats}')
            pbar.update(train_state.cur_step - prev_step)
            prev_step = train_state.cur_step

    stats = measure_game_stats_against_random_player(
        key, init_fn, step_fn, train_state.policy_net
    )
    all_game_stats.append(stats)
    print(f'Step {train_state.cur_step}: {stats}')

    return policy_net, all_game_stats


if __name__ == '__main__':
    train_model()
