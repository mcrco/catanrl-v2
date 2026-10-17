"""catanllm's prompting strategies, vendored (attribution: Marco Yang's catanllm,
``~/dev/catanllm/prompts``, SP 25 CS 159).

Each strategy is a system prompt: a preamble that fixes the answer format
(reasoning, then a one-line note the player keeps for later turns, then the
action number alone on the last line), for ``strategy`` and ``rules`` a guide
or the rulebook, and the board coordinate explanation. catanllm's files
hard-code the BASE board's tile corners and node neighbours; here that
section is generated from the map the game is played on (``board_section``),
and the preamble states the game's own victory-point target (base-2p plays to
15, not 10). ``plan`` is catanllm's mini-map prompt without its fixed board
(the state text carries the board). Texts are Python strings, not package
data, so ``code_id`` (provenance) hashes them: a prompt edit is an outcome
code change.
"""

from __future__ import annotations

from typing import Any

STRATEGIES = ("basic", "strategy", "search", "rules", "plan")

#: catanllm/prompts/basic_prompt.txt up to the board section ({vps} was "10").
BASIC = """\
You are an expert Settlers of Catan player. Your goal is to win the game by reaching {vps} victory points first.
You should think about which action to take every single time. Explain your reasoning, step by step.
However, your output should always end with these two lines:

A brief summary of the reasoning behind your move. You will use these as notes when making future moves. Output these notes in one line with no header. 
The action number (The line should only be the number.)

Do not put a hyphen, bullet point, or any other formatting before these two final lines.
"""

#: catanllm/prompts/search_prompt.txt up to the board section.
SEARCH = """\
You are an expert Settlers of Catan player. Your goal is to win the game by reaching {vps} victory points first.
You should think about which action to take every single time. Explain your reasoning, step by step.
In your reasoning, you should simulate a tree search on moves that you think will bring you the most benefit by considering the consequences of each action.
However, your output should always end with these two lines:

A brief summary of the reasoning behind your move. You will use these as notes when making future moves. Output these notes in one line with no header. 
The action number (The line should only be the number.)

Do not put a hyphen, bullet point, or any other formatting before these two final lines.
"""

#: catanllm/prompts/mini/plan_prompt.txt: what it adds after the basic preamble.
PLAN_EXTRA = """\
In your step-by-step reasoning, you should weight the benefits of each action and plan future actions as well. For example, if you have the resources for a road, but you are very close to being able to build a settlement, you should hold off on building the road and save your resources for the settlement instead by choosing to end your turn immediately. Also, factor in your opponent's potential moves and how you can thwart their strategies. Make sure to include your plan based on these considerations in the brief summary.
"""

#: catanllm/prompts/strategy_prompt.txt: the guide between the preamble and the board section.
STRATEGY_GUIDE = """\
=== ADVANCED STRATEGY GUIDE TO WIN MORE CATAN ===

Use the below guide when reasoning about your strategy.

Here are 17 TIPS AND TRICKS from the New York City Champion — so you earn more bragging rights.

* *General Strategy*
* *Placements*
* *Trading*
* *Table Presence*
* *Miscellaneous Tactics*

Yes, there really is a competitive scene for Settlers of Catan and we all take it *way too seriously*.

Performing data analysis on hundreds of matches, debating even the most insignificant of moves, and recapping games like what you see in poker or chess — there’s no end to what some people like me will do to gain an edge.

Given you’re here — I commend you! You and me are alike — we HATE losing in Settlers of Catan. Other games I can come in last place and not care the next minute — this one…not so much.

If you’re sick of being on the bad end of trades without even realizing it and find yourself winning until it’s stripped away from you at the very last moment — you’ve come to the right place.

Reading this guide will legitimately pay dividends for years to come. Whether you have family game night later today or are playing a decade from now, I have no doubt a few pointers you are about to learn will stick with you.

This post is designed to have each pointer as a stand-alone principle to keep in mind. Thus, feel free to jump around and revert back to this post for some last-minute refreshers before playing a game.

**Who am I you ask?** As someone who has been studying Catan for years and even won a couple tournaments — I have learned a lot of tricks of the trade, inventing some myself. For further background on my story, check out my first post here: “my strange addiction: settlers of catan”.

Think of this guide as learnings from a thousand games culminating into one short list. Inevitably, sharing my unique approach will take away some of my edge; however, candidly, I want to grow on Medium more — so if you get any value from this, follow my Medium for additional Catan content.

I know writing a piece as lengthy as this one will drop off the reading completion rate than if I wrote something half the size, but ultimately I decided that there’s a lot to say and those whom I would really like to help improve their strategy will read through anyways. This took a lot of effort, but I really enjoyed writing. There are a lot of gold nuggets in here *even if you may not agree with everything* — I hope you enjoy!

Alright, let’s get into the juicy stuff already. Keep in mind, this is an “advanced” guide so not ideal for those who just learned the rules and excludes a lot of the most basic strategies. *(If you still need that cheat sheet on what cards to build a city — this means you)* On the flip side, some of these tips are not revolutionary for the experienced player, but worth keeping top of mind.

Several of these principles, especially regarding *table presence* (more on that later), probably don’t matter against your 3 friends who are each 7 White Claws deep trying to watch sports at the same time.

These rules of thought are most effective at a table of competent players who are dialed-in. Most often, you’ll find this at tournaments rather than at any given house game with little-to-no stakes.

Additionally, since some of these tactics are a bit cold-blooded *(doing whatever it takes to win)*, don’t expect everything included herein to keep your competition happy — user discretion is advised. Even though Catan is just a game and you should be able to do what’s best to win, people do take moves personally. **Sometimes keeping people happy at the expense of short-term wins builds goodwill that could pay off down the line.**

Anyways, let’s get started…

**1. Know your competition**

* This is what I would deem to be the most important field of thought throughout any match: **knowing who you’re playing against at any point in time**. Specifically, understanding personalities, strategic prowess, and temperament. Each of these vectors are subject to change throughout a game and can drastically shape your path to victory. Everyone’s headspace is different whether they’re far in the lead…or not. It is your responsibility to recognize this shift so you can capitalize accordingly. Since Catan is so interpersonal, **realizing how each person may work *with* or *against* everyone else matters a lot.**
* There’s this inherent balancing act of how assertive vs. passive someone should be toward who and when. In the game of life, overbearing people usually lose, but in Catan..this is not always the case. Outspoken players may either influence others not to double-cross them or can inadvertently paint a large target on their backs — it depends.
* Since Catan is so interpersonal, **how you go about proposing, negotiating, and rejecting trades is critical.**

2. Bluffing

* This is the area of Catan that really separates the kids from the adults. It is openly framing to the table that you’re going to do something even when you have no plans to. *(Not to be confused with going back on a deal)*

**Here are some examples:**

* Persuading someone to build their road a certain way claiming you’re going to place a settlement there — meanwhile you intend to keep it open as an expansion spot.
* Extorting someone to trade with you so that you won’t knight them, even when you never planned on putting the robber on them at all. *(note: this may have since been outlawed in the official tournament Catan rules)*
* Declaring you have a specific development card (such as a knight) so that someone won’t do an action (like putting the robber on you).
* Negotiating a more attractive trade to help save the game. I’ll explain: imagine you need to steal the longest road to save the match. Even if you have the sufficient cards to make this happen, you can declare that you’re not going to build the roads unless you get an even more-than-necessary deal from the table. (Of course, if the others aren’t aware of this — no need to proclaim it, simply state that you need help) This declaration that you’re willing to let the game end, even when you’re not, can be quite effective. I would recommend going as far as pretending you are about to hand over the dice (even though you won’t finish the motion) to get someone else to fold first to your trade demands.
* Even when you really want a trade, it’s often helpful to sound indecisive of how much it may benefit you. Coming across eager will often make the other person recoil their willingness to follow through with it. It also may build some goodwill with the person if you say the trade is helping them more than you and to keep it in mind when you are offering a deal next time.
* Further to needing a specific card and acting indifferent, it may be best to offer the card(s) you want to give away in exchange for “anything”. This can help obscure the move you would like to make when you only accept the trade you want. If someone offers something useless to you, correct yourself that you don’t want that resource. If you’re unsure of what cards are in circulation as well, it will give you an idea of what people have (or more specifically, what cards people don’t want).
* The list goes on…

3. Cut people off, but don’t let it happen to you

* There are three main stages to any round of Catan: the early-, mid-, and end-game. Each one is defined by it’s unique priorities, with the early-game focused on expansion (i.e. building roads). Securing the settlement spots left unclaimed from initial placements should always be your first priority (building cities is best to focus on later). The rationale is that you will not only gain more production from *different* hexes, but also diversification of both *resources and numbers* (*see Tip #3*).

**Cut people off**

* Even if you are two or three roads away from cutting someone off, don’t let this deter you. If you are fairly confident you may be able to build somewhere before someone else, focus on the contested spot(s) first. You may derail someone else’s game who positioned their settlements assuming they could get it — a great offensive move by you. Meanwhile, you are also gaining a spot that may be accretive to your production. Anywhere else that may be less desirable could still be up for grabs later too.
* When cutting someone off, it’s best to build all the required roads on *one single turn*. Otherwise, if you build one road to start (even if it helps get you to 7 or less cards), the other player will see what you’re doing and prioritize getting there before you. Thus, you will have wasted your resources building to nowhere (if there is no back-up route).

**Don’t get cut off**

* Especially during the early stage of the game, AVOID getting cut off at all costs. Pay close attention to anyone who threatens to steal the spot you’re aiming for. Keep track if they are accumulating sufficient lumber/brick, be cognizant of any trades that may help them on this objective, and assume any unturned development card could be a year of plenty, or even worse, a road builder. If there is a mutually beneficial trade possible, you can sweeten the deal for yourself by having their promise not to build toward you. *(While verbal agreements like this may not be ‘binding’ per Catan rules, they are almost always upheld)*
* Even while initial placements are being made, try to dissuade players from pointing their roads in your direction. Even if it’s a spot that is unlikely for you to get anytime soon, it helps to give you optionality later on. I will often proclaim that they are building toward a high priority spot of mine that would be better for the both of us if we did not have to compete for it.

4. Avoid being the clear leader

* **Early-to-mid game:** You will get knighted to oblivion and miss out on trades if you get too much of a lead early. This is why those who often start strong aren’t able to maintain it. If you have access to ample ore and grain, it may be more appropriate to buy development cards than cities. I have found that development cards are often more subtle. It doesn’t raise alarm bells while bringing you closer to earning the largest army or gaining another beneficial card.
* **End game:** If the largest army or longest road is still up for grabs and someone else is not about to take either — delay getting it for yourself. For example, if you have two knights face-up with at least one face down (assuming nobody else is in the same boat), it may be worth holding out on flipping it. (The exception is that by not flipping it, you may encourage someone else to buy development cards in order to catch-up to you or be forced to play a knight at an inopportune time) Being close to {vps} VP’s places more of a target on you. When possible, get those 2 VP’s during your move to win the game.

5. Diversify your resources *and* number production

* **Resource diversity:** It’s* typically* better to have access to more (or all) of the five resources. Thereby, you are less reliant on trading a 4–1 with the bank, 2–1 with a port, or forced into making a deal with someone.
* **Number diversity:** You want to have your numbers as distributed as possible. For example, while the numbers 6 and 8 have equal probabilities of being rolled, it is better to have access to a 6 AND 8, rather than two 6’s or two 8's.
* *Here’s why:* your production is less concentrated on isolated numbers. Therefore, if a needed roll of the dice doesn’t come, the effect on you will be less drastic and variable. With staggered production (aka accessing repeat numbers), you are more likely to hold more than 7 cards or few in your hand at any point.

6. How to play your development cards

* **Road builder:** To maximize it’s impact, use this to cut someone off toward a desirable spot or make an unexpected steal of longest road to win the game. Alternatively, if your production of lumber/brick is minimal so you’re having a tough time building roads, it is equally fine to use road builder for acquiring an uncontested settlement spot.
* **Year of plenty:** Unless you need to steal the longest road or urgently need to acquire a development card to hold the largest army, use this card for the resources you are producing least. For example, if you have access to ore on your hexes, but need one or two so you can build a city now — don’t. Wait until those numbers are rolled organically instead.
* **Monopoly:** This card increases in power as the game progresses. When production is much higher since there are more cities in play, you will get more bang for your buck.
* **Knight:** Even without using it, you can threaten others that if the robber is placed on you, you will use the knight against them as revenge (whether or not this is a bluff — is up to you). Additionally, in times when your hand is “stuck” (you need another card to make a move on your turn) and the robber is not on you, it may be fine to use the knight simply for taking a card from someone.

7. Robbers can decide games

* New players often underestimate the power of the robber. It can drastically slow down someone’s game, even if it may just *block one number* or *steal one card* at a time. If it’s placed on you — okay, fine — but really avoid being stolen from as well. The robber deserves respect and caution.

**Here are some pointers:**

* If there’s an obvious placement that does not involve you and the person may not realize, it may be best to bring it to their attention. Otherwise, it’s better to keep a low profile to let them arrive at the decision and not direct attention to you.
* Direct your attention away from the game itself. This means avoiding eye contact with the player robbing someone and potentially chatting up someone uninvolved. And yes, it really can work. Someone who is indifferent will take a card from the person who is more engaged in what’s happening — it’s more convenient and potentially won’t have to interrupt your conversation.
* If the person does choose to take from you nonetheless, you can tell them that you don’t have the resource they need *(even naming said resource if you know what they want)*. You can even ask them and then share their odds of actually getting it — sharing their better odds with the other person, if you’re really paying attention.
* Keep your cards close to your body when someone is deciding who to pick from. Remember that note about making it convenient for them? If they want to pick from you, make them reach — sometimes people will actually change their minds during this moment.
* Most notably, always try to make a trade if it makes sense to either have someone not knight you or at least take a card from you. It’s often better to trade a card *(unless it’s a resource you really need)* for something you don’t want than to lose one.
* When in control of the robber, you must also keep in mind the opposite perspective of everything above. When you may be placing the robber, use these opportunities to extort specific cards out of other players in exchange for you not knighting/taking from them.

8. Always speak up

* Remember: every trade you are not involved in is helping the games of others, not yours.
* If there is a trade that may be at least slightly one-sided, bring this light so the deal does not occur. You want to highlight how much the exchange will help the aggressor and only be a marginal benefit for the other. You can even offer a different deal including yourself. While every situation is different, Catan does not typically favor those who remain silent.

9. Think about win-scenarios

* One of the most common scenarios people lose is that they have tunnel vision on their own win-scenarios (the path someone needs to win) while losing sight of everyone else.
* In most games, the winner will have the longest road or largest army — therefore, if you’re too far from acquiring one of them, you need to try to secure the other. As you think about which each person is poised to go after based on their resource production (lumber/brick = road, ore/grain/wool = largest army), don’t ever lose focus on what it would take for them to secure it. And if they do, their proximity to winning.
* It’s difficult for someone to sneak a win with largest army, but the longest road is a different story. The fastest Catan matches usually end with the longest road ripe for the taking at 5-roads or the one holding it having not insulated theirs to be long enough.

10. The person with the most points isn’t always winning

* Pay attention to who is most likely to win the game long term (i.e: strength of resource production, open spots for future settlements, ability to build roads or buy development cards). In matches with beginners, players will often incorrectly target someone who may get an early longest road or otherwise get to 5 points quickly. However, slowing the person with the best long-term set up and highest skill level early is often best. In these instances, their games may accelerate beyond what can be stopped.

11. Yes, 11’s and 3’s roll

* Even at the highest level of Catan, I have found that some people do not pay much attention to these numbers. They each have a **5.6% probability of rolling** — meaningful odds that should not be counted out. They may not be ideal for your initial settlement placements, but could make for great expansion numbers. Not to mention, they will likely never have the robber on them.

12. Table presence

* Your mannerisms at the table matter a lot and are contingent on the field of thought that started this post: knowing your competition. There are times it may be better to keep a low profile while at other moments it may best to be outspoken so nobody wants to point the target on you.
* There are tons of nuances for how you should behave in a game based on the unique circumstances at hand. For example, whether or not you should have urgency for someone to accept a trade should vary based on who you may be playing against — people react very differently.
* Furthermore, how you choose your words matters a lot during negotiations. If you are desperate for a resource, do not make this clear to the other player — use the word “*want*” instead of “*need*”. If you show weakness, the person will capitalize and get a better deal from you.
* When playing against strangers (with stakes while in a tournament), it often pays to get on their good side. This includes *acting* remorseful when robbing from them and *faking* empathy when their numbers don’t roll. Half joking, but also half serious. If you want to be next-level try hard, you can even make some off topic conversation while playing, getting their heads off the game and for them to like you.
* My favorite example of this strategy was when I was able to portray whatever table presence I wanted during the preliminary round of the NYC tournament I won in December 2022. Sitting with three strangers, it seemed most advantageous for me to act timid and come across like I barely knew the rules. I kept asking for advice on moves that I should make which helped to keep the target off me the entire game — even when I was winning. One guy (who ended up making the semi-finals) even openly said that he felt bad knighting me because I was so nice. Ultimately, I ended up winning and shared that I had played in a few tournaments before.

13. Card tracking

* **At the highest level of play, tracking is what separates the great players from the best.** This is the act of keeping a tally of who has what resources. This is a tremendous advantage because you are awake of what trades someone is willing to make so that you can maximize every exchange. It also keeps you informed on what move someone is focused on making next.
* While at first this may seem unrealistic, it becomes easier over time the more you practice. As the game progresses and more trades are made, knowing exactly what cards everyone has is impossible. However, a general idea of what someone has can be sufficient. This is done by remembering the previous rolls that have occurred and what someone has spent their resources on. If they must discard cards from having more than 7, pay attention to what they get rid of.
* When first learning to track cards, start with a reasonable goal of keeping track of the last 5 rolls at any given point. Therefore, even if you don’t remember what each person has in their hand, you can back solve what cards each person has based on where they are positioned on the board.

14. Involve yourself in others’ initial placements

* While many people will view the start of the game as independent (where there shouldn’t be any dialogue) — this could not be farther from the case. Sometimes there are obvious settlement spots/road directions to start a game; however, most of the time there is a level of subjectivity that you can take advantage of.
* If there are two spots of equal interest to someone, it’s worth guiding them toward the one that interferes with you less. Additionally, you want to mitigate situations when a player points their road toward a spot you both may then have to compete for. There could be another spot or road placement genuinely better for them that may not entail taking the same level of risk.

15. Keep a low profile

* When nearing the lead, the less unnecessary attention you can draw to yourself, the better. While these points may seem dramatic, I assure you they can make a difference against people who really care to win.
* If the bank’s resources are in stacks across the table from you, ask to split the piles in half on each side or have everything moved closer to you. Constantly reaching across the board for cards will accentuate how much you’re picking up.
* If you’re picking up 3 cards of the same resource, pick all 3 up at once instead of 1 at a time to not accentuate all the cards you’re getting.
* When the robber is placed on you and someone is deciding who to pick a card from: always keep your cards in one stack instead of fanned out so that you’re not flaunting how many cards you have. If you have more than a few, this may be the reason someone may pick from you.

16. Picking up development cards

* Just like in poker, do not give a ‘read’ on what development card you’ve picked up. Instead, when you buy them, do not look and keep them in front of you facedown. Only look at them at some later point in the cycle. Your expression, even more than you realize, can signal what you’ve picked up and play a big role in hurting your game.
* Naturally, the exception to this is when you’re pulling for a VP to win it all.
* On the flip side, closely watch the expressions of opponents when they look at their new development cards.

**If they…**

* glance at the card quickly without any reaction, then place it down — maybe a knight.
* have a look of distaste being early in the game — maybe a Victory Point.
* look around to see what cards everyone has — maybe a monopoly.
* pay closer attention to the board — maybe a road builder on where they may go.
* look back at their hand to see what they need — maybe a year of plenty.

17. Directing trades

* Cerberus paribus, it’s better to be on the side offering trades than accepting them. Unless you know exactly what’s in someone’s hand and what they will do or whom they may try to trade with, a move may happen that you’re prepared for.
* In Catan, you will regret accepting someone’s trade much more often than you will be happy you did it. A willingness to walk away from deals is the highest form of leverage.
"""

#: catanllm/prompts/rules_prompt.txt: the rulebook between the preamble and the board section.
RULEBOOK = """\
=== CATAN RULEBOOK ===

**THE GAME: CATAN**

**INTRODUCTION**
Welcome to the unexplored island of Catan!
As you and your fellow explorers gaze across the uninhabited landscape of hexagonal tracts of terrain, you quickly realize that this unique land arrangement provides ample space for each of you to put down roots.

You race off to establish your respective communities by gathering resources from nearby terrain, building your infrastructure, and nurturing trade relationships. Watch out for surprises-- your rivals can block your path or send the robber to plunder your wealth!

Good luck, explorers!

**OBJECTIVE**
The first player to reach {vps} victory points (VPs) on their turn wins. Earn VPs by building. Collect and trade for the resources you need to build.

* Settlement: 1 VP each
* City: 2 VPs each
* Largest Army and Longest Route: 2 VPs each
* Development card with a VP: 1 VP each

**CONTENTS**
* 6 sea frame pieces
* 19 terrain hexes (3x hills, 4x forests, 4x pastures, 4x fields, 3x mountains, 1x desert)
* 2 bonus victory point (VP) tiles: Longest Route and Largest Army
* 4 player aids.
* 18 number discs (2: 1x, 3: 2x, 4: 2x, 5: 2x, 6: 2x, 8: 2x, 9: 2x, 10: 2x, 11: 2x, 12: 1x)
* 25 development cards (2x Monopoly, 2x Road Building, 2x Invention, 14x Knight, 5x Victory Point)
* 95 resource cards (19x brick, 19x wood, 19x wool, 19x wheat, 19x ore)
* 2 card trays
* 2 dice
* 1 robber
* 20 settlements (5x each color)
* 16 cities (4x each color)
* 60 roads (15x each color)

Each player aid reads the following:
**PLAYER AID**
**TURN OVERVIEW**
**Production Phase**
* Play a development card (optional)
* Roll dice
* Collect resources or Resolve a 7

**Action Phase**
* In any order (optional):
* Trade
* Build roads, settlements, cities, and development cards
* Play a development card (if you did not play one before the dice roll)

**PLAY DEVELOPMENT CARDS**
* Not on the turn you bought it
* Only 1 development card per turn
* Before rolling dice or during the Action phase
* VP card exception: You may play multiple VP cards (even on the turn you buy them) in order to win the game.

If this is your first time playing CATAN, we recommend using the setup shown here. For additional variety, check out the Variable Setup described on page 11.

**1. Assemble the Frame**
Match the numbers at the puzzle piece ends of the sea frame pieces to assemble the coast of Catan.

**2. Place the Hexes and Number Discs**
Place the terrain hexes and number discs inside the frame as shown.

**3. Create the Supply**
Sort the resource cards by type and place them in five faceup stacks in the card trays. Shuffle the development cards and place them in a facedown stack in the empty card tray slot. Place the Longest Route and Largest Army tiles near the game board.

**4. Place the Robber and Player Pieces**
* Place the robber on the desert hex
* Each player selects a color and takes the roads and buildings (settlements and cities) in that color along with a player aid. In a 3-player game, do not use the white pieces.
* Place 2 starting settlements and roads for each player as shown.

**5. Collect Your Starting Resources**
Each player takes the resource cards from the supply that match the hexes adjacent to their second settlement, highlighted in black. Keep these cards hidden in your hand.

**6. Choose the First Player**
Each player rolls the dice. The player with the highest roll is the first player.

**TURN OVERVIEW**
CATAN is played over a series of turns, starting with the first player, and moving clockwise around the table. A turn consists of two phases, which occur in this order:
1. Production phase
2. Action phase
Once you have finished the Action phase, if you have not won, pass the dice to the player on your left. They begin their turn with the Production phase.

**PRODUCTION PHASE**
**PLAY A DEVELOPMENT CARD**
If you have development cards, you may play one of them before rolling the dice. See "Development Cards" on page 9.
**ROLL DICE**
Roll both dice and add them together. This total determines which hexes produce resources this turn.
**COLLECT RESOURCES**
Hexes with a number disc matching the dice roll produce this turn. Each player with a settlement on a producing hex receives 1 resource card of the corresponding type from the supply.

If a player has 2 or 3 settlements on that hex, they receive 1 resource card for each settlement.

Similarly, a player receives 2 resource cards for each of their cities on that hex.

If there are not enough resource cards in the supply to fulfill everyone's production, then no one receives any of that resource. However, if only one player is affected, give that player as many of those resource cards as remain in the supply.

* Forests produce wood
* Hills produce brick
* Pastures produce wool
* Mountains produce ore
* Fields produce wheat
* The desert produces nothing

**Example:**
The White player rolls an 8. Blue has a settlement on the mountain hex with the number 8 disc, so Blue receives 1 ore resource card. Orange receives 2 ore for their city on that hex.

**RESOLVE A 7**
When you roll a 7, hexes do not produce any resources. Instead:
**1. DISCARD RESOURCES**
Each player who has more than 7 resource cards must choose half (rounded down) of their resource cards and return them to the supply.

**Example:**
White has 9 resource cards in hand. When a 7 is rolled, they must discard 4 cards.

**2. ACTIVATE THE ROBBER**
You must move the robber to a new hex.

Steal 1 random resource card from a player who has a building on that hex. If multiple players have buildings on the new hex, you choose one player to rob.
**Important:**
A hex with the robber does not produce resources when its number is rolled.

**Example:**
Orange rolls a 7. They move the robber to the pasture hex with a red settlement and a blue city. Orange chooses to steal a card from Blue and takes a random card without looking.

**ACTION PHASE**
You may take actions in this phase as often as you like and in any order, as long as you have the resources to do so.

**TRADE**
You may trade freely with other players and the supply to get the resources you need to build. During your turn, other players may only trade with you, not with each other or with the supply.
There are three types of trades you may perform:

**TRADE WITH OTHER PLAYERS**
To trade with other players, announce which resource(s) you want and which resource(s) you are willing to trade. Other players may accept your proposal, make counteroffers, or make their own proposals.
**Important:**
You may not give away cards in any way, which includes trading matching resource cards (for example, trying to trade 3 ore for 1 ore is not allowed).
**Example:**
It is Blue™s turn, and they want 1 wheat. Orange will trade 1 wheat and wants 1 wool in return. Blue agrees and the players exchange cards. White also wants to trade for 1 wool, but Blue no longer has any. White may not trade with Orange because it is not Orange's turn.

**GENERAL TRADE WITH THE SUPPLY (4:1)**
To trade with the supply, put 4 of the same resource cards into the supply and take 1 card of a different resource from the supply.
**Example:**
It is Blue's turn. They choose to trade 4 wood for 1 ore.

**PORT TRADE WITH THE SUPPLY (3:1 OR 2:1)**
If you have a building on a 3:1 port, you may put 3 of the same resource cards into the supply and take 1 card of a different resource from the supply.

If you have a building on a 2:1 port, you may put 2 of the resource cards shown on the port into the supply and take 1 card of a different resource from the supply.

**Example:**
White wants 1 ore. On their turn, they may not trade 2 wheat for 1 ore because they do not have a building on the 2:1 wheat port. However, they may trade 3 wood for 1 ore because they have a settlement built on a 3:1 port.

**BUILD**
Roads, buildings (settlements and cities), and development cards each have different costs. During your turn, you may spend resources (aka "build") to place roads and buildings on the board, as well as to take development cards from the supply. To do so, simply return the required resource cards from your hand to the supply.

**ROADS**
Roads are worth 0 VP
Roads are placed on empty hex edges. A new road must connect to one of your existing roads or buildings. You may not build a road starting on the other side of an opponent's building.
Building cost: 1 wood, 1 brick

**Longest Route:** the first player to have 5 continuous roads in play receives this tile. If another player has more continuous roads in play, they immediately receive this tile. The Longest Route tile is worth 2 VPs.
If, when a player's route is broken, they no longer meet the requirements for the Longest Route, the tile is returned to the supply. It remains in the supply until a single player has the longest continuous route of at least 5 roads. That layer immediately receives the bonus tile and the 2 VPs.
**Example:**
Red has a continuous route of 7 roads and the Longest Route tile. White has a continuous route of 6 roads. On White™s turn, they build a settlement at the intersection shown, breaking Red™s route into a four-segment route and a three-segment route. Since White now has the longest route, they receive the tile and the 2 VPs.

**SETTLEMENTS**
Settlements are worth 1 VP.
Settlements are placed on empty intersections. A new settlement must follow the Distance Rule and must connect to at least one of your existing roads. You have 5 settlement pieces. To continue building more settlements, you must first upgrade one to a city.
Building cost: 1 brick, 1 wood, 1 wool, 1 wheat

**The Distance Rule:**
When placing a settlement, stay at least two edges away from all other buildings!

**CITIES**
Cities are worth 2 VPs.

Cities always replace settlements. To build a city, remove one of your settlements from the board, return it to your player area, and place your city where the settlement was located. You have 4 cities and may not build more.
Building cost: 2 wheat, 3 ore

**DEVELOPMENT CARDS**
When you build a development card, draw the top card of the deck. Development cards stay hidden until used. They do not count toward the number of cards in your hand when a 7 is rolled, and the robber may not steal them.
You may play 1 development card during your turn by placing it face up in your player area. It may not be a card you built this turn. You may play a development card before rolling dice or at any time during the Action phase.
Development cards never go back into the supply. If the supply runs out, you may not build any more cards.
You may not trade or give away development cards.
Building cost: 1 wheat, 1 wool, 1 ore

**Types of development cards:**
**INVENTION**
When you play this card, take any 2 resource cards from the supply. Add them to your hand. They may be 2 of the same resource of 2 different resources.

**MONOPOLY**
When you play this card, announce one type of resource. Each player must give you all their resource cards of that type. You may only request one type of resource when you play this card, regardless of how many cards you receive.

**ROAD BUILDING**
When you play this card, build 2 roads at no cost (i.e. without spending any resource cards).

**VICTORY POINT**
Reveal all your Victory Point cards, including any built this turn, if you can reach the number of VPs needed to win. Otherwise, you must keep Victory Point cards hidden in your player area.

**KNIGHT**
Move the robber to a new hex. Steal 1 random resource card from a player with a building on that hex.

**Largest Army:** The first player to have 3 Knight cards in play receives this tile.
If another player has more Knight cards in play, they immediately receive this tile.
The Largest Army tile is worth 2 VPs.

**WINNING THE GAME**
If you have {vps} or more VPs at any point during your turn, the game ends immediately and you are the winner!
Turn over any number of Victory Point cards, including ones built this turn, to show that you have reached {vps} VPs.

Congratulations!
**Example (the base game, played to ten points):**
Blue ends the game on their turn with ten VPs (2 cities, 3 settlements, the Longest Route tile, and 1 Victory Point card). Victory!
"""

#: catanllm's user-message instruction (util/llm_wrapper.py).
INSTRUCTION = ("Based on the current game state and available actions, provide a brief explanation of your "
               "reasoning and choose the best action by responding with the action number (e.g., '2' for action 2).")

#: What the model is told when its answer could not be used (a re-prompt, then the fallback).
RETRY = ("Your previous answer could not be used: {error}. Answer again. Your output must end with the "
         "one-line notes and then the action number alone on the last line.")

SPATIAL = """\
--- SPATIAL RELATIONSHIPS OVERVIEW ---
- Tiles: Hexagonal tiles, identified by cube coordinates (x,y,z) where x+y+z=0
- Nodes: Intersection points between tiles, identified by integer IDs (0-{last_node})
- Edges/Roads: Connections between two nodes, represented as (node_id1, node_id2)
- Each tile has 6 nodes at its corners and 6 edges connecting those nodes
- Adjacent tiles share nodes and edges
- Ports are located on specific edges facing the ocean
"""


def board_section(catan_map: Any, *, intro: bool = False) -> str:
    """catanllm's ``=== BOARD COORDINATE SYSTEM EXPLANATION ===`` for ``catan_map`` (a
    Catanatron ``CatanMap``): every land tile's corner nodes, every land node's
    neighbours (the map's own graph, land only) and adjacent tiles."""
    lines = ["=== BOARD COORDINATE SYSTEM EXPLANATION ===", ""]
    if intro:
        lines += ["Below is an explanation for how the board/game state will be represented in natural language.",
                  ""]
    lines += SPATIAL.format(last_node=max(catan_map.land_nodes)).splitlines()
    lines += ["", "--- HEXAGON TILES ---"]
    for coordinate, tile in sorted(catan_map.land_tiles.items()):
        lines.append(f"Tile {tuple(coordinate)} (ID {tile.id}): corner nodes: {sorted(tile.nodes.values())}")
    lines += ["", "--- NODES ---"]
    for node in sorted(catan_map.land_nodes):
        neighbors = sorted(catan_map.graph.neighbors(node))
        tiles = ", ".join(str(t.id) for t in sorted(catan_map.adjacent_tiles.get(node, []), key=lambda t: t.id))
        lines.append(f"Node {node}: neighbors: {neighbors} | adjacent to tile IDs: {tiles}")
    return "\n".join(lines) + "\n"


def system_prompt(strategy: str, catan_map: Any, *, vps: int = 10) -> str:
    """The system prompt of ``strategy`` for a game on ``catan_map`` played to ``vps`` points."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; one of {STRATEGIES}")
    if strategy == "search":
        return SEARCH.format(vps=vps) + "\n" + board_section(catan_map)
    basic = BASIC.format(vps=vps)
    if strategy == "basic":
        return basic + "\n" + board_section(catan_map)
    if strategy == "plan":
        return basic + "\n" + PLAN_EXTRA + "\n" + board_section(catan_map)
    if strategy == "strategy":  # the guide's and the rulebook's "10 VP" sentences state this game's target too
        return basic + "\n" + STRATEGY_GUIDE.format(vps=vps) + "\n" + board_section(catan_map, intro=True)
    return basic + "\n" + RULEBOOK.format(vps=vps) + "\n" + board_section(catan_map)
