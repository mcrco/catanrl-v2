#import "../macros.typ": *

= monte carlo playouts instead of a tree <sec:playouts>

marco's idea: replace the search tree with plain playouts. for the policy's top k actions, play n full games from each with the raw policy on sampled worlds, and take the action with the best win rate. costed from our own numbers, it is far too slow as a player at k = 5, n = 1,000 (\~150 gpu hours a 1,000-game row), but cheap enough as a reference that grades the search and the critic. nothing is queued.

=== monte carlo playouts over the top actions #chip("todo") <exp:playouts>
#meta(
  dates: "2026-10-23",
  dates-label: "proposed",
  owner: "marco",
  wandb: (),
  records: ("831eea84",),
)
#tldr[costed, not run. as a player at k = 5, n = 1,000 it would cost \~150 gpu hours a 1,000-game row, against 3.9 h for mcts at 2048 sims, so not at that size. it is feasible as a reference q^π at a few hundred positions, or as a cheaper player with sequential halving or truncated playouts (both my suggestions, not queued).]

*idea.* k ≈ 5 actions, n ≈ 1,000 playouts each, no critic at all. the policy plays fair (it sees only its own observation), so a playout in a world dealt from the tracker is a proper sample of the outcome. that makes this one step of policy iteration (tesauro's backgammon rollouts). the engine has the pieces: `cpp.sample_world` deals a world from a seat's tracker, and games copy.

*cost.*
#figure(
  btable(
    columns: (1fr, auto),
    align: (left, right),
    header: ([quantity], [value]),
    [decisions in a 2p game (`831eea84`)], [\~159, 81 of them the searching seat's],
    [decisions left from a typical decision], [\~80],
    [policy steps per decision at k = 5, n = 1,000], [\~400k],
    [ppo rollout speed on the 2070 super], [\~42k decisions/s],
    [time per decision / per game], [\~10 s / \~10 min],
    [one 1,000-game harness row], [\~150 gpu h (mcts at 2048 sims: 3.9 h)],
    [one decision on the bot vm], [minutes],
  ),
  caption: [the cost of playouts as a player, from our own numbers.],
)

*status.* not viable as a player at that size. two cheaper uses:
+ *a reference.* q^π(s, a) for the top actions at a few hundred positions (500 positions ≈ 1.5 gpu hours), with common random numbers (the same worlds and dice for every action): ±1.6 pp per action at n = 1,000. it would grade the search's choices and the critic far better than one noisy 8192-sim search. (my suggestion, not queued)
+ *a cheaper player.* sequential halving over the actions, or truncated playouts (d moves, then the critic), at maybe 5–10× the cost of mcts at 2048. (my suggestion, not queued)
