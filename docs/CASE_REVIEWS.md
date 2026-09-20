# Five case reviews from the selected submission (`final_J_bags20_ev18.csv`)

Each review lists the pair, its risk rank in our submission, the five submitted evidence hands, what is observable in the log, and a plausible benign explanation. Cards are as recorded in `seats.parquet`; "P1/P2" are the pair's `player_1`/`player_2` from `evaluation_pairs.csv`. All amounts are chips; blinds are given per hand.

---

## Case 1 — `P8FA9939F1886` (rank 1, predicted `directed_transfer`)
Players `U73C3EEF1BEAB` (P1) / `UA25B1D20ECFE` (P2), 99 shared evaluation hands.
Evidence: `HDBA527F2C2D221`, `HD9EB30F28B5458`, `HA08CD42774F1F5`, `H950EF8646547F1`, `HE8DFAF2396BF2B`.

**Observable behaviour.** In every listed hand P1 commits a large share of the stack to P2 with a hopeless holding, and P2 is the sole beneficiary.
* `HDBA527F2C2D221` (bb 2): P1 calls from the big blind with 8♥3♥, check-calls the flop, bets 46 on the turn with 8-high (board 6♦4♥T♠4♣), calls P2's raise to 208, then shoves the river for 78 into P2's 7♥7♠. P1 loses the entire 299 stack; P2 +304.
* `HD9EB30F28B5458` (bb 2): P1 calls P2's 3-bet with Q♠7♠, then calls 27, 89 and an all-in on 5♣4♠K♥T♦T♣ with queen-high against K♦J♠. P1 −240.
* `HA08CD42774F1F5` (bb 2): P1 calls two re-raises pre-flop (12, then 95) with 4♥3♦ behind P2's raises, and folds the flop to P2's shove. P1 −109, P2 +157.
* `H950EF8646547F1` (bb 2): P1 bets and then shoves three streets with A♦2♠ into P2's A♠9♠ on Q♠A♣6♣T♠3♣; P1 −170.
* `HE8DFAF2396BF2B` (bb 2): P1 calls a raise with 6♣5♥, check-folds the flop to P2.
Over the 99 shared hands, 526 big blinds move from P1 to P2 and 148 the other way (net P1 −378 bb, P2 +387 bb). With P2 at the table P1 enters 33 % of pots and 20 % of hands with junk cards; without P2, 24 % and 6 % — P1's play against outsiders is ordinary.

**Plausible benign alternative.** P1 could simply be a very loose, bluff-heavy player who happens to run into P2's strong hands: the 8-high turn bet and river shove in `HDBA527F2C2D221` are the kind of bluff a reckless player makes against anyone, and A2 vs A9 (`H950EF8646547F1`) is a common dominated-ace cooler. Individually, each hand is explainable as bad play; it is the repetition, the one-directional flow, and the contrast with P1's play against outsiders that make coordination the more likely reading.

---

## Case 2 — `PAE6DC4BF00FE` (rank 9, predicted `soft_play`)
Players `UB56EF3A5E19A` (P1) / `UB8D99E5B936F` (P2), 119 shared hands.
Evidence: `H715D63EA5BFD96`, `H49C4C83639A21E`, `H0466DD13AAD8F6`, `HF013A54C234011`, `HF3315B7996E692`.

**Observable behaviour.** Heads-up against each other, P1 declines to bet made hands and folds them to small bets.
* `H715D63EA5BFD96` (bb 4): P1 limps 9♠9♥ (a hand the population raises), flops a **set of nines** on 5♦T♦9♣, checks the flop and turn heads-up against P2, and folds the river to a 2-big-blind bet. P2 wins with T♣8♥.
* `H49C4C83639A21E` (bb 4): P1 raises K♠9♠, flops a pair of nines on 5♥8♥Q♠, checks three streets and folds to P2's 20-chip river bet; P2 holds J♥A♦ (no pair).
* `H0466DD13AAD8F6` (bb 4): P1 and P2 check every street to showdown (2♦2♥3♥5♠T♥); P2's 8♣8♠ wins a pot that never grows.
* `HF013A54C234011` (bb 4): P1 calls P2's bets on all three streets with T♦J♠ (second pair on K♥Q♣5♣5♠T♥) and pays off 266 to P2's A♣Q♠.
* `HF3315B7996E692` (bb 4): multi-way; the partners check together on the flop and both fold to an outsider.
Pots between the partners either stay minimal (mutual checking, folding sets and pairs) or, when P2 bets, P1 calls down and loses; P1 never wins a meaningful pot from P2 in the shared hands.

**Plausible benign alternative.** P1 may be a tight-passive player who slow-plays sets and pairs and folds to any river aggression regardless of opponent, and two passive players will often check hands down. The set-of-nines fold for two big blinds is extreme but is a single hand; the check-downs (`H0466DD13AAD8F6`) are common between passive players and would not be suspicious on their own.

---

## Case 3 — `P063AEE30B6A8` (rank 8, predicted `coordinated_isolation`)
Players `U0D9E9B605C76` (P1) / `UD45A5F2A67BC` (P2), 95 shared hands.
Evidence: `H9F48475CB65CE2`, `H7D5E4AF2B28B0E`, `HC33F4F889C651D`, `HBD0544815956A5`, `H4F2E56049F11BB`.

**Observable behaviour.** The pair raises and re-raises each other pre-flop with junk to push outsiders out, then one partner folds to the other.
* `H9F48475CB65CE2` (bb 4): P1 opens 3♦4♣ to 9, P2 3-bets K♣T♥ to 27, an outsider 4-bets A♥K♦ to 70; both partners fold.
* `HC33F4F889C651D` (bb 4): P1 opens A♦6♠ to 12, P2 3-bets **9♦2♦** to 36, the four outsiders fold, P1 folds. P2 wins 18.
* `HBD0544815956A5` (bb 4): P1 opens 6♠2♦, P2 3-bets 8♦2♣ to 28, an outsider calls, P1 4-bets to 90 with 6-2 offsuit, P2 folds, the outsider calls and bets the flop; P1 folds (−98).
* `H7D5E4AF2B28B0E` (bb 4): P2 opens 7♦3♣ to 12 and takes the blinds.
* `H4F2E56049F11BB` (bb 4): P1 opens A♠J♣, P2 folds, an outsider re-raises twice and P1 folds.
Three-bets and four-bets with 9-2, 8-2 and 6-2 between the same two players, followed by a fold to the partner once outsiders are gone, recur throughout the shared hands.

**Plausible benign alternative.** Two hyper-aggressive players who 3-bet and 4-bet light against every opener would produce the same raise-fold cycles, and `HBD0544815956A5` in fact costs the pair 126 chips to an outsider, which is what you would expect from reckless aggression rather than coordination. The pattern is only suspicious because it is concentrated on hands where the two are in the pot together.

---

## Case 4 — `P8698759D2395` (rank 250, predicted `coordinated_isolation`)
Players `U3EF56F211745` (P1) / `U52952C699E3B` (P2), 50 shared hands.
Evidence: `HBC05929C497F6F`, `H79C01885E6B1A0`, `H9BE917B77C8489`, `HCA7E700AFB342E`, `H5E3E4AF27D7877`.

**Observable behaviour.** Both partners open-raise very weak hands and re-raise each other pre-flop.
* `HBC05929C497F6F` (bb 2): P2 opens 9♥5♣ to 5 and takes the blinds.
* `H5E3E4AF27D7877` (bb 2): P1 opens T♠3♦ to 4 and takes the blinds.
* `H9BE917B77C8489` (bb 2): P2 opens Q♠7♥; an outsider 3-bets and both partners fold.
* `H79C01885E6B1A0` (bb 2): P2 opens J♥9♠, P1 3-bets 8♥8♣, everyone including P2 folds.
* `HCA7E700AFB342E` (bb 2): P2 opens A♦7♦, P1 3-bets A♠K♥, an outsider calls, P2 4-bets to 30 with A-7, the outsider 5-bets 8♥8♣, P1 shoves, the outsider calls, P2 folds. P1 loses 200 to the outsider's 9♦4♦ (a rivered flush on Q♥6♠5♦T♦J♦); P2 −36.

**Plausible benign alternative.** With only 50 shared hands this is a weak case, and our model ranks it accordingly (250th). Opening 95o/T3o/Q7o and 4-betting A7 suited are consistent with two ordinary loose-aggressive players; the isolation attempt in `HCA7E700AFB342E` backfires against a third player, which is at least as consistent with independent aggression as with coordination.

---

## Case 5 — `P1944BCEB32C8` (rank 450, predicted `directed_transfer`)
Players `U4715F363D43C` (P1) / `U69D0BEF5871D` (P2), 160 shared hands.
Evidence: `H7B727FBC6EBAB2`, `HB613D24BED4EF7`, `HF3C572F01DFD65`, `H888B9D10B6DD99`, `HCE7BA51652C427`.

**Observable behaviour.** P2 repeatedly enters pots against P1 with junk and calls large bets with nothing; P1 holds a strong hand each time.
* `H7B727FBC6EBAB2` (bb 2): P2 opens T♠6♥, calls P1's 3-bet, then calls 25, 29 and 94 with ten-high on 3♥4♣A♦3♠7♠ against K♥K♣. P2 −161.
* `HB613D24BED4EF7` (bb 2): P2 calls with 3♠T♥, calls a flop check-raise to 76 with ten-high on A♦4♣8♦, bets the turn, and calls the river all-in; loses the whole 138 stack to P1's pair of fours.
* `HF3C572F01DFD65` (bb 2): P2 calls with 7♣8♦, calls flop and turn, then raises the river to 86 on 3♥4♥4♦3♠4♠ (playing the board) into P1's A♣A♦. P2 −113.
* `H888B9D10B6DD99` (bb 2): P2 calls down three streets with 4♦6♦ (pair of sixes) against P1's A♦8♥ on 6♣7♣Q♣T♦9♦.
* `HCE7BA51652C427` (bb 2): P2 calls a 3-bet with J♦3♥, bets the flop and folds to P1's raise (−33).
Over the 160 shared hands 325 big blinds move from P2 to P1 and 18 the other way. P2 calls 0.31 times per hand when P1 is dealt in versus 0.18 without P1; the junk-entry rate is similar in both settings (20 % vs 18 %), which is why the contrast is weaker than in Case 1.

**Plausible benign alternative.** P2 could be a calling-station bot that pays off anyone with any holding; every one of these hands is an ordinary bad call, and P1 simply had the goods. The model's lower rank (450) reflects that the with-partner versus without-partner contrast is weaker here than in Cases 1–3.
