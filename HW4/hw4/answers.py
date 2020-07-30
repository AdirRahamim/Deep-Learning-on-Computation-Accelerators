r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.99,
              beta=0.5,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=16,
              gamma=0.95,
              beta=0.2,
              learn_rate=1e-3,
              eps=1e-8)
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=64,
              gamma=0.99,
              beta=1.,
              delta=0.005,
              learn_rate=8e-3,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=16,
              gamma=0.99,
              beta=0.5,
              delta=1.0,
              learn_rate=1e-3,
              eps=1e-8,
              )
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**
Subtracting the baseline from the reward in the policy gradient results in the advantage, the measure of how much the
current action is better than what we had usually do in that state. This helps reduce the variance since the advantage
has lower variance because the baseline compensates for the variance introduced since its not depand on the state.
An example where it helps is in complicated tasks, where the policy may receive very different rewards for similar states,
by subtracting the baseline we reduce the variance and by using a lot of experience  to average over the different rewards
, due to the low variance it will eventually converge to good behaviour.
"""


part1_q2 = r"""
**Your answer:**
$v_{\pi}(s)$ equals to the expectation of $q_{\pi}(s,a)$ over all possible actions, i.e $v_{\pi}(s)=\sum_{a \in A} q_{\pi}(s,a) \cdot p(a_0=a)$
Where $q$ is expectation over all possible trajectories given initial action and state.
Its not possible to compute all possible trajectories from all possible states and all possible actions, thus we use the
$q$ value as the baseline. That way we get a good approximation for $v_{\pi}$ while we run the regression.
"""


part1_q3 = r"""
**Your answer:**
1. We can notice several differences between the graphs. It seems that using the baseline made the training less noisy,
with less jumps, mostly in the beginning, the reason is because of the decrease in the variance as I explained before.
Bpg and cpg achieved same baseline loss at the end, however bpg converged little faster. The reason might be the entropy
loss as minimizing the max entropy loss means we want our agent try to explore more and not always go on the safe side.
In the loss_e graph after about half of the episodes cpg achieved lower loss, thus we can infer that using baseline
indeed help in the training process.
In the loss_p graph cpg and bpg(both using baseline) look like a constant line with jitters because we estimate the 
average for every batch, and the estimation is close to the average by definition.
In the mean_reward graph they all seem to achieve very close results, with some picks of improvement for cpg.

2. In the loss_p graph we can see that AAC achieved about much lower loss, and he converged much faster.
For the loss_e graph, we can see AAC had about the same loss but agian it converged much faster.
In the mean reward we can see AAC has much less noisy graph, it converged faster and have much higher reward than all
other graphs.
 """
