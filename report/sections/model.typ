#import "../macros.typ": *

= Search in a learned model <sec:model>

Marco's idea (2026-11-16): MuZero in place of AlphaZero, with a model that predicts the state one or two dice rolls ahead, so that the search looks deeper for less. Something we want to try; nothing is built or queued.

=== A learned model that jumps one or two dice rolls #chip("todo") <exp:model-jump>
#meta(
  dates: "2026-11-16",
  dates-label: "proposed",
  owner: "marco",
  wandb: (),
  records: (),
)
#tldr[Grow the search tree in a learned model instead of the engine, one step per dice roll instead of one per decision. Our numbers say depth is dear and the tree's cost is on the CPU, so there is room; the open risks are the other seats' moves folded into a jump, and legal moves without rules. Proposed first stage: measure offline whether a jump can be predicted at all.]

*Idea.* A tree node is a vector the network made, not a copy of the game. One model step goes from a decision of the searching seat to the position one or two dice rolls later, skipping what lies between. The engine still plays the real games; only the search uses the model.

#lead[What our numbers say.]
- *Depth is dear today.* 1,024 sims look 2.1 dice rolls and 7.5 decisions ahead; 131,072 sims look 3.75 rolls and 13 decisions (@exp:search-noise). A step of one roll would reach those horizons in two to four steps.
- *More search still pays.* Each 4× in sims adds 3–7 points up to 16,384 (@exp:search-depth-2p).
- *The tree's cost is on the CPU.* Every node holds a copy of the game; freeing finished trees is 19% of self-play's wall-clock, and an L40S sits idle 80% of the time (@exp:az-hardware). A tree of vectors lives on the GPU and copies no games. A simulation still costs a network pass, so the card's work per simulation does not fall.

#lead[What is hard.]
- *Chance.* Dice, card draws and steals need the stochastic form of MuZero (Antonoglou et al., 2022): the model learns a small set of chance outcomes and their odds. It matched AlphaZero, which searches the real rules, in backgammon and 2048; it did not beat it. A gain here has to come from speed or from the jump.
- *The other seats inside a jump.* A jump over an opponent's turn folds their moves into the model as what they usually do, not their best reply. The search then plans against an average opponent and finds no refutations. Catan is mostly one's own income and builds, so this may cost little; not measured. In 3p and 4p a jump skips up to three seats, where today's tree is widest (4p search gains 2 points, @exp:search-4p).
- *Legal moves.* The model has no rules: the legal set inside the tree is a prediction, and its errors compound with depth.
- *Balanced dice.* The odds of the next roll depend on the deck, so the model must read the dice input (@exp:dice-input).
- *Training cost.* MuZero from scratch needs more games than AlphaZero, because it learns the model too. Here it would start from the AlphaZero net and add the model, trained on the same self-play games.

#lead[Stages (Claude's proposal, not queued).]
+ *Can a jump be predicted?* No search. Record self-play games of the current net as sequences and train only the model on them. Measure, one and two rolls ahead and given what the dice and the other seats did, how well the predicted position gives the real net's value and move at the real later position. A recording tool, a model head and a few GPU hours. If two rolls ahead predicts poorly, stop here.
+ *Search in the model against search in the engine.* The same net, the same held-out games, equal time per move.
+ *Self-play with it.* Only if stage 2 wins.

*Cost.* Stage 1 is small and could run on Modal beside other work (@exp:recipe-modal). Stages 2 and 3 need a new search on the GPU, the larger part of the work.
