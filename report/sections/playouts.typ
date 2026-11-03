#import "../macros.typ": *

= Monte Carlo playouts instead of a tree <sec:playouts>

Marco's idea: replace the search tree with plain playouts. For the policy's top _k_ actions, play _n_ full games from each with the raw policy on sampled worlds, and take the action with the best win rate. Costed from our own numbers, it is far too slow as a player at _k_~=~5, _n_~=~1,000, but cheap enough as a reference that grades the search and the critic. Nothing is queued.

=== Monte Carlo playouts over the top actions #chip("todo") <exp:playouts>
#meta(
  dates: "2026-10-23",
  dates-label: "proposed",
  owner: "marco",
  wandb: (),
  records: ("831eea84",),
)
#tldr[Costed, not run. As a player at _k_~=~5, _n_~=~1,000 it would cost \~150 GPU hours a 1,000-game row, against 3.9 h for MCTS at 2,048 sims. It is feasible as a reference $Q^pi$ at a few hundred positions, or as a cheaper player with sequential halving or truncated playouts.]

*Idea.* _k_ ≈ 5 actions, _n_ ≈ 1,000 playouts each, no critic at all. The policy plays fair (it sees only its own observation), so a playout in a world dealt from the tracker is a proper sample of the outcome. That makes this one step of policy iteration (Tesauro's backgammon rollouts). The engine has the pieces: `cpp.sample_world` deals a world from a seat's tracker, and games copy.

#lead[Cost.]
#figure(
  btable(
    columns: (auto, 1fr),
    align: (left, left),
    header: ([Quantity], [Value]),
    [Decisions in a 2p game (`831eea84`)], [\~159, 81 of them the searching seat's],
    [Decisions left from a typical decision], [\~80],
    [Policy steps per decision at _k_~=~5, _n_~=~1,000], [\~400k],
    [PPO rollout speed on the RTX 2070 Super], [\~42k decisions/s],
    [Time per decision / per game], [\~10 s / \~10 min],
    [One 1,000-game harness row], [\~150 GPU h (MCTS at 2,048 sims: 3.9 h)],
    [One decision on the bot VM], [Minutes],
  ),
  caption: [The cost of playouts as a player, from our own numbers.],
)

*Status.* Not viable as a player at that size. Two cheaper uses:
+ *A reference.* $Q^pi (s, a)$ for the top actions at a few hundred positions (500 positions ≈ 1.5 GPU hours), with common random numbers (the same worlds and dice for every action): ±1.6 pp per action at _n_~=~1,000. It would grade the search's choices and the critic far better than one noisy 8,192-sim search. (Claude's suggestion, not queued)
+ *A cheaper player.* Sequential halving over the actions, or truncated playouts (_d_ moves, then the critic), at maybe 5–10× the cost of MCTS at 2,048 sims. (Claude's suggestion, not queued)
