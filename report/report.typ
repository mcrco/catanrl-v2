= basic arch ablations

previously did some informal arch ablations, and realized that the graph network proposed in canopy is the fastest, most efficient, and most generalizable (can scale to arbitrary maps since you predict board actions per intersection/road/tile). but would still be interested in doing a formal ablation with MLP, LSTM, CNN (like in the old catanrl), and GNN (canopy) eventually

= current best recipe

implement engine in c, turn into pufferlib env, run for ~1B steps, and get a really good shared-trunk critic + policy that beats `ValueFunctionPlayer` ~90% of the time

put the critic + policy inside a mcts harness, with a simple probabilistic belief-sampling determinization method to get another ~5% h2h edge over the raw policy.

insert wandb training logs here.

= exploring ideas

- training alphazero (currently agreement is really high, doesn't seem to yield many gains)
- training one model to train for up to `n` players and arbitrary maps (should be possible by using masked values for numeric features and GNN already handles arbitrary hex boards)
- adding trading to the engine
- allowing the model to "learn training"
  - do not increase action space explosively, either
  - have the model predict the player trade action and a float for each of the 5 types of elements, for both what they give and what they get (11 extra options in the action space), and round to the nearest integer for each of the 10 floats, or
  - have the model predict the player trade action and a float for each of the 18 possible integer choices for each of the 5 types of elements, for both what they give and what they get (181 extra options)
  - i suspect that training while allowing this from scratch will likely not result in the model learning to trade well. instead, we should first train a competent model using the ppo recipe, then take random game states, try applying all every possibe legal trades to the current state, and evaluate which trades modify the game state such that the new game state is better for both sides of the trade, according to the critic model
- build the best possible full-information tracker for any number of players that resolves hidden information (e.g. a robber steal between 2 other players) if either of the other players reveals what happened in the trade via a new action
  - the only hidden information are dev cards and robberies.
  - for example: let us be player A. at the start, we know player B's hand is a wood and a sheep. player C's hand is a brick, an ore, and a wheat. if player B robs player C, and then buys a road, we can resolve that player B now has a sheep, while player C has an ore and a wheat. or if player C robbed player B, and then bought a dev card, we know that player C has a brick, and player B has a wood.
  - another example: let us be player A. we have 6 knights played, player B has 6 knights played, and player C has 1 knight played. then, if player C plays another knight, we know that there are no more knights in anyone else's hand, or in the development card deck, because player C has just played the 14th (and last) knight. we should then never account for any knights in the game. this is important because say if player B has largest army, we know we should stop trying to contend for largest army because we can't get buy any more knight cards. obviously this goes way further than this, since we should be updating our prior on our belief set for the sampling determinization method every time someone plays a dev card.
- train the model to predict the full-information observation from the public-information observation instead of sampling from belief
- define different scopes for how "markov" a model decision is
  - right now, we only feed the model the latest view of the game state as its input
  - we could instead train a sequential model on the entire game state history
    - e.g. transformer or lstm
  - this would allow the model to learn things like "if someone didn't play a knight 2 moves ago when one of their high production tiles was blocked, then they probably don't have a knight among their dev cards"
  - this uses many orders of magnitude more compute, but would be only way to get "perfect" bot
- turn this into an LLM RL env/eval (e.g. harbor task) and RL a small language model on it, and eval frontier models on how well they play catan
- turn this task of making a strong catan bot an LLM RL env/eval, and there is a good chance the next frontier models will become amazing at creating better catan bots.
