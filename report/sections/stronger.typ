#import "../macros.typ": *

= A stronger 1v1 bot: inputs, heads and search <sec:stronger>

Marco (2026-11-15), before more AlphaZero training: think about ways to make the network or the search more intelligent, such as the dice shuffle as an input and changes to MCTS that fit Catan, and read #link("https://benburk.ca/catan-law-of-small-numbers/")[a blog post on Colonist's balanced dice]. Six ideas came out of that. Marco approved all six in outline and this order: measure what limits the search (@exp:search-noise), then the dice state as an input (@exp:az-dice). The other four wait on those two.

== Where the bot stands

=== The OpenHex ladder and the balanced-dice post #chip("done") <exp:stronger-context>
#meta(
  dates: "2026-11-15",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Our best bot is 11th of 42 rated 1v1 bots, about 80 Elo from the top, and the leader's name says it plays its network's top move without search. The post's dice model is our engine's, constant for constant; our search already uses the exact odds, but the network has no dice input.]

#lead[Ladder.]
#figure(
  btable(
    columns: (auto, 1fr, auto, auto),
    align: (right, left, right, right),
    header: ([Rank], [Bot], [Elo], [Won / played]),
    [1], [2s-NN-Medium-Greedy], [1703], [724 / 1,107],
    [2], [Gaz], [1696], [89 / 155],
    [3], [Ham], [1684], [11 / 16],
    [4], [Pattern], [1681], [70 / 128],
    [5], [2s-NN-Mini-1024], [1675], [611 / 1,090],
    [6], [oversized-pig: "Neural evaluation with sampled lookahead."], [1675], [236 / 377],
    [7], [puma-ppo-864m], [1672], [31 / 55],
    [8], [2s-NN-Medium-256], [1658], [694 / 1,068],
    [9], [HexN: "Graph network, 64 wide, 332k parameters. Search by paired rollouts over the top 6 moves, 64 worlds, 8 turns deep."], [1655], [629 / 973],
    [10], [2s-NN-Mini-256], [1649], [699 / 1,236],
    [11], [`catanrl-1v1-search-v3` (e300 in the search; @exp:deploy-openhex-search-v3)], [1623], [122 / 220],
    [12], [2s-NN-Mini-Greedy], [1610], [754 / 1,266],
    [14], [`catanrl-1v1-search-v2` (fr3 in the search)], [1571], [241 / 464],
    [16], [`catanrl-1v1-search` (an older net, 1,024 sims)], [1559], [726 / 1,444],
  ),
  caption: [OpenHex's 1v1 ladder on 2026-11-15 (public records; 42 bots with rated 1v1 games). Quotes are the bots' own descriptions.],
) <tab:stronger-ladder>

- *The network is our weaker half.* The leader and the 12th bot are raw networks by their names ("Greedy"); a sibling's description says "3x1024 ResBlock ... greedy argmax policy, 0 MCTS sims". Our raw fr3 bot sits at 1557. On the ladder our search added about 16 Elo to fr3 (1557 to 1571, different dates and opponents).
- *What the others search with.* HexN compares its top six moves on the same 64 sampled futures (common random numbers, @exp:search-paired-root); none describes a tree as deep as ours.
- `search-v3.1` (e1800, @exp:deploy-openhex-search-v3-1) went live the same afternoon (bot `e296048b5a1c4a83`; 3 of its first 6 games won, too few to rate).

#lead[The post.] "Exploiting the law of small numbers in Catan" reverse-engineers Colonist's balanced dice: a 36-card deck reshuffled when 13 cards are left, a penalty on totals rolled in the last five rolls, and in 1v1 a multiplier on sevens that evens out each player's share and breaks streaks. A player who tracks all of it puts about 20% more probability on the roll that comes (0.26 bits a roll).
- *Same dice.* `engine/cppanatron/include/cppanatron/dice.hpp` has the post's constants: refill at 13, a window of 5, 0.34 per recent copy, a seven multiplier clamped to 0–2 with ±0.4 per streak step.
- *Who uses them here.* The search's chance nodes draw rolls from the exact odds, and so do its sampled worlds. The network sees none of it: its value and its prior are the same whatever the deck holds.
- *How much the odds move* (our dice-only simulation, `tools/dice_odds_sim.py` on `wt/dice-input`): the roller's P(7) runs 4–28% (fair 16.7%), and on 58% of rolls some total cannot come.
- *The post's advice* (track the deck, the window and the seven share; place the robber and hold cards by them) is what a network with the dice state as input could learn (@exp:az-dice).

== Measure first

=== What limits the search: seed agreement and depth in dice rolls #chip("running") <exp:search-noise>
#meta(
  dates: "2026-11-15",
  dates-label: "started",
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[Claude's suggestion, approved by Marco on 2026-11-15. Two measurements on e1800 in the deployed search, to say whether the search is short of samples or short of a better network: do two searches of one position with different seeds pick the same move, above all when they overrule the network, and how many dice rolls deep do the simulations reach?]

*Question.* Is the deployed search limited by sampling noise or by the network's values?

#lead[Why.] A dice node draws its outcome at random from the exact odds (`sample_outcome`), and the search plays one sampled world. A search of 1,024 simulations spread over 11 totals and two or three rolls of depth has few samples per branch. If noise decides the move, the search ideas pay (@exp:search-stratified, @exp:search-paired-root); if two seeds agree and the tree is shallow, the network is the limit (@exp:az-dice, @exp:az-aux, @exp:az-graph-m).

#lead[Setup.]
- *Positions.* Decisions with at least two legal moves from new games of e1800 against fr3 on `base-2p-fr` (seed 20261115, not a suite's), spread from the opening placements to the endgame, at most two per game.
- *Seed agreement.* Each position is searched twice with different search seeds, in the deployed options (one world, 64 pending, transpositions, fused kernels) on a fresh tree: 4,000 positions at 1,024 sims (the evals' depth) and up to 1,000 of them at 131,072 (the bot's). Reported with Wilson intervals, split by opening placement and main game: how often the two agree, how often the search overrules the network's top move, and when it does, whether the second search makes the same move, the network's, or a third.
- *Depth.* Over all simulations, the number of dice rolls and of decisions between the root and the leaf.
- No games are played and no win rate comes out: this is a diagnostic, about 2.5 h of GPU.

*Status.* Being built and run on the GPU (`wt/search-diag`, `tools/search_noise.py`).

== The network

=== The dice state as an input, in the AlphaZero line #chip("queued") <exp:az-dice>
#meta(
  dates: "for the GPU after the search measurement",
  dates-label: "queued",
  owner: "marco",
  config: "configs/experiments/az-2p-dice.yaml",
  wandb: (),
  records: (),
)
#tldr[Marco's idea: keep track of the shuffled cards and pass that to the model. Two continuations of e1800 under its own recipe, identical except that one network reads the dice state. The control arm is the continued training Marco asked for anyway.]

*Question.* Does a network that sees the balanced dice state make a stronger search than the same network without it?

#lead[Setup.]
- *Input* (off by default; wider than the 22 next-roll odds built on 2026-11-10 and never run). For the whole board: each seat's odds for the next roll, the cards left in the deck per total, the last five rolls, the draws left before the refill, and each seat's sevens and the current streak. On each tile, for its own number: its odds on my roll and on the opponent's, its cards left and its count in the window. Every new weight starts at zero, so at e0 the `dice` arm plays exactly as e1800.
- *Arms.* `control` and `dice`, both from `az-2p-kata/main+1500@e1800` (`2p-graph-s-6`) with the recipe of @exp:az-kata-1800, 600 iterations each (about 14 h; about 110,000 games). Both start with a fresh optimizer and an empty replay: e1800's stored positions have no dice state, so `dice` cannot inherit them, and `control` starts the same way to keep the pair fair.
- *Evals.* No selects. One held-out row per arm at the end: the last checkpoint in the 1,024-sim search against `BELIEF(fr3, 16 worlds)` on 2,500 `heldout-2p-fr` boards × 2 seats, the two arms paired game for game (McNemar). At 5,000 games the paired difference has a standard error of about 0.8 pp, so about 2.1 pp is detectable at 80% power; a smaller edge could go unseen. Each row takes about 5.8 h, so evals are about 29% of each arm's 20 h.
- *Free readouts.* Both arms' losses on W&B, and in `dice` the new weights' norm and a probe: on a fixed batch, how far the policy and the value move between the true dice state and a fresh deck.

*Status.* Being built on `wt/dice2` from `wt/dice-input`. Sizes are Claude's proposal; Marco can change them before launch.

=== Auxiliary heads: each seat's final VP and the turns left #chip("todo") <exp:az-aux>
#meta(
  dates: "2026-11-15",
  dates-label: "proposed",
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[Claude's suggestion, approved in outline by Marco on 2026-11-15; not queued. KataGo trains its trunk on more than win or loss: ownership and score heads give many labels per position and are dropped at play time (about 1.65× training speed in its ablation). Catan's analogs: each seat's final VP and the number of turns left.]

A game gives the value head one bit. Final VP and game length are known for every stored position at no extra self-play cost, and the graph network already has an auxiliary head (`GraphAuxHead`, @sec:hidden). It would run as a third arm of @exp:az-dice's shape, after that pair.

=== `graph-m` in the AlphaZero line #chip("todo") <exp:az-graph-m>
#meta(
  dates: "2026-11-15",
  dates-label: "proposed",
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[Approved in outline by Marco on 2026-11-15; not queued. `graph-m` beats `graph-s` as a raw policy (@exp:arch-graph-m-2p-lr), and the ladder says our network is the weaker half. It costs 3.4× as much per leaf and the fused kernels do not cover its width yet, so it comes after the cheaper ideas.]

== The search

=== Stratified dice nodes #chip("todo") <exp:search-stratified>
#meta(
  dates: "2026-11-15",
  dates-label: "proposed",
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[Claude's suggestion, approved in outline by Marco on 2026-11-15; not queued. At a dice node, visit the total furthest below its share of the visits instead of drawing one at random. The value of the node is the same weighted mean with less noise at equal simulations. A change to one function (`sample_outcome`); built if @exp:search-noise says the search is noise-limited.]

=== Paired candidates at the root #chip("todo") <exp:search-paired-root>
#meta(
  dates: "2026-11-15",
  dates-label: "proposed",
  owner: "assistant",
  wandb: (),
  records: (),
)
#tldr[Claude's suggestion (HexN's approach, @tab:stronger-ladder), approved in outline by Marco on 2026-11-15; not queued. Judge the root's top moves on the same worlds and the same dice, so that the luck of the draw cancels in their difference. Built if @exp:search-noise says the search is noise-limited and stratified dice nodes do not close the gap.]
