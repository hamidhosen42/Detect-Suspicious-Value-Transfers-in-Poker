"""
Hypothesis `ev_transfer`: evidence hands are the hands with the largest EXPECTED-VALUE transfer between partners
(not the largest chip transfer). All six hole cards are known, so per-street equity of every player against any
subset of still-active opponents is computed (exact enumeration on flop/turn/river, Monte Carlo preflop), and each
partner decision is scored as the EV it surrenders / gives to the partner under a one-street pot-odds model.

USAGE
  python3 work_step3/hyp/ev_transfer.py                 # dev: features for the 45,129 positive-pair hands + evaluation
  python3 work_step3/hyp/ev_transfer.py --phase eval    # eval: features for every (pair_id, hand_id) in work_step3/eval_pair_hands.parquet
  python3 work_step3/hyp/ev_transfer.py --phase eval --pairs my_pairs.parquet   # any table with pair_id, hand_id, player_1, player_2

HOW TO COMPUTE THE SAME FEATURES FOR EVALUATION PAIRS
  build_features(pairs_df) needs only (pair_id, hand_id, player_1, player_2) plus data/hands.parquet, data/seats.parquet,
  data/actions.parquet. Nothing in the feature layer touches labels, evidence files or behaviour families:
    1. equity_tables(hand_ids)         -> eq[h, street, seat, active_mask]  (numba; MC preflop with fixed per-hand seed)
    2. action_frame(hand_ids)          -> one row per action with the active mask before the action, last aggressor, equities
    3. pair_hand_features(actions, pairs) -> per (pair_id, hand_id) EV-transfer aggregates (chips are in big blinds)
    4. within-pair percentile ranks (`*_prank`) are computed over all hands of the pair (same as the pipeline's pranks)
  For the eval phase pass the eval (pair_id, hand_id, player_1, player_2) table; the equity step is ~1.5 ms/hand/thread.
  Run with NUMBA_NUM_THREADS=3 (set below) to stay inside the shared-machine budget.

LEAKAGE: is_ev / evidence_rank / labels are used ONLY in the evaluation section (AUC, MAP@5 harness, spearman diagnostic).
"""
import os, sys, time, argparse, warnings
os.environ.setdefault("NUMBA_NUM_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np, polars as pl
from numba import njit, prange
warnings.filterwarnings("ignore")

ROOT = "/Users/md.hamidhosen/Documents/Kaggle/Detect Suspicious Value Transfers in Poker"
D = f"{ROOT}/data/"; W = f"{ROOT}/work_step3/"; H = f"{W}hyp/"
RANK_CHARS = "23456789TJQKA"; SUIT_CHARS = "cdhs"
CARD_MAP = {f"{r}{s}": i * 4 + j for i, r in enumerate(RANK_CHARS) for j, s in enumerate(SUIT_CHARS)}
N_PF = 1000  # preflop Monte Carlo rollouts (5 board cards from the 40 unseen cards); flop/turn/river are exact

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

# ----------------------------------------------------------------------------------------------------------------------
# 1. direct 7-card evaluator (same value encoding as the pipeline: category*15^5 + tie-breakers)
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _straight_high(mask):
    for hi in range(12, 3, -1):
        if (mask >> (hi - 4)) & 31 == 31:
            return hi
    if (mask & 0b1000000001111) == 0b1000000001111:
        return 3
    return -1

@njit(cache=True)
def eval_n(cards, n):
    """Best 5-card value from the first n (5..7) cards of `cards` (ints rank*4+suit)."""
    B = 15; B5 = 759375
    rc = np.zeros(13, np.int64); sc = np.zeros(4, np.int64); smask = np.zeros(4, np.int64)
    rmask = 0
    for i in range(n):
        c = cards[i]; r = c >> 2; s = c & 3
        rc[r] += 1; sc[s] += 1; smask[s] |= 1 << r; rmask |= 1 << r
    # flush / straight flush
    fs = -1
    for s in range(4):
        if sc[s] >= 5:
            fs = s
    if fs >= 0:
        sh = _straight_high(smask[fs])
        if sh >= 0:
            return 8 * B5 + sh
    four = -1; three = -1; three2 = -1; p_hi = -1; p_lo = -1
    for r in range(12, -1, -1):
        if rc[r] == 4:
            four = r
        elif rc[r] == 3:
            if three < 0:
                three = r
            elif three2 < 0:
                three2 = r
        elif rc[r] == 2:
            if p_hi < 0:
                p_hi = r
            elif p_lo < 0:
                p_lo = r
    if four >= 0:
        k = -1
        for r in range(12, -1, -1):
            if rc[r] > 0 and r != four:
                k = r; break
        return 7 * B5 + four * B + k
    if three >= 0 and (p_hi >= 0 or three2 >= 0):
        p = p_hi if p_hi > three2 else three2
        return 6 * B5 + three * B + p
    if fs >= 0:
        v = 5 * B5; k = 0; m = smask[fs]
        for r in range(12, -1, -1):
            if (m >> r) & 1:
                v += r * B ** (4 - k); k += 1
                if k == 5:
                    break
        return v
    sh = _straight_high(rmask)
    if sh >= 0:
        return 4 * B5 + sh
    if three >= 0:
        k0 = -1; k1 = -1
        for r in range(12, -1, -1):
            if rc[r] > 0 and r != three:
                if k0 < 0:
                    k0 = r
                elif k1 < 0:
                    k1 = r; break
        return 3 * B5 + three * B * B + k0 * B + k1
    if p_hi >= 0 and p_lo >= 0:
        k0 = -1
        for r in range(12, -1, -1):
            if rc[r] > 0 and r != p_hi and r != p_lo:
                k0 = r; break
        return 2 * B5 + p_hi * B * B + p_lo * B + k0
    if p_hi >= 0:
        v = B5 + p_hi * B ** 3; k = 0
        for r in range(12, -1, -1):
            if rc[r] > 0 and r != p_hi:
                v += r * B ** (2 - k); k += 1
                if k == 3:
                    break
        return v
    v = 0; k = 0
    for r in range(12, -1, -1):
        if rc[r] > 0:
            v += r * B ** (4 - k); k += 1
            if k == 5:
                break
    return v

# ----------------------------------------------------------------------------------------------------------------------
# 2. equity of every seat vs every active subset, per street
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _xorshift(state):
    x = state
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFFFFFFFFFF

@njit(cache=True)
def _accumulate(V, acc):
    """V[6] final values of the 6 seats for one board; acc[6,64] += win share of seat p within active mask m (m contains p)."""
    mx = np.empty(64, np.int64); cnt = np.empty(64, np.int64)
    for m in range(1, 64):
        low = m & (-m)
        idx = 0
        while (1 << idx) != low:
            idx += 1
        rest = m ^ low
        if rest == 0:
            mx[m] = V[idx]; cnt[m] = 1
        else:
            if V[idx] > mx[rest]:
                mx[m] = V[idx]; cnt[m] = 1
            elif V[idx] == mx[rest]:
                mx[m] = mx[rest]; cnt[m] = cnt[rest] + 1
            else:
                mx[m] = mx[rest]; cnt[m] = cnt[rest]
    for p in range(6):
        vp = V[p]
        for m in range(1, 64):
            if (m >> p) & 1 and mx[m] == vp:
                acc[p, m] += 1.0 / cnt[m]

@njit(cache=True)
def _street_equity(hole, board, nb, street, eq_out, seed):
    """Fill eq_out[6,64] with equity of each seat vs each active subset at `street` (0..3) given board[:nb]."""
    known = np.zeros(52, np.bool_)
    for p in range(6):
        known[hole[p, 0]] = True; known[hole[p, 1]] = True
    nfix = 0 if street == 0 else (3 if street == 1 else (4 if street == 2 else 5))
    for i in range(nfix):
        known[board[i]] = True
    deck = np.empty(52, np.int64); nd = 0
    for c in range(52):
        if not known[c]:
            deck[nd] = c; nd += 1
    acc = np.zeros((6, 64), np.float64)
    cards = np.empty(7, np.int64); V = np.empty(6, np.int64); full = np.empty(5, np.int64)
    for i in range(nfix):
        full[i] = board[i]
    nroll = 0
    if street == 0:
        state = seed * 2654435761 + 12345
        state &= 0xFFFFFFFFFFFFFFFF
        if state == 0:
            state = 88172645463325252
        for r in range(N_PF):
            # sample 5 distinct cards from deck[:nd] by partial Fisher-Yates
            for j in range(5):
                state = _xorshift(state)
                k = j + (state >> 11) % (nd - j)
                t = deck[j]; deck[j] = deck[k]; deck[k] = t
                full[j] = deck[j]
            for p in range(6):
                cards[0] = hole[p, 0]; cards[1] = hole[p, 1]
                for j in range(5):
                    cards[2 + j] = full[j]
                V[p] = eval_n(cards, 7)
            _accumulate(V, acc); nroll += 1
    elif street == 1:
        for a in range(nd):
            for b in range(a + 1, nd):
                full[3] = deck[a]; full[4] = deck[b]
                for p in range(6):
                    cards[0] = hole[p, 0]; cards[1] = hole[p, 1]
                    for j in range(5):
                        cards[2 + j] = full[j]
                    V[p] = eval_n(cards, 7)
                _accumulate(V, acc); nroll += 1
    elif street == 2:
        for a in range(nd):
            full[4] = deck[a]
            for p in range(6):
                cards[0] = hole[p, 0]; cards[1] = hole[p, 1]
                for j in range(5):
                    cards[2 + j] = full[j]
                V[p] = eval_n(cards, 7)
            _accumulate(V, acc); nroll += 1
    else:
        for p in range(6):
            cards[0] = hole[p, 0]; cards[1] = hole[p, 1]
            for j in range(5):
                cards[2 + j] = full[j]
            V[p] = eval_n(cards, 7)
        _accumulate(V, acc); nroll += 1
    for p in range(6):
        for m in range(64):
            eq_out[p, m] = acc[p, m] / nroll

@njit(cache=True, parallel=True)
def equity_all(hole, board, nboard, max_street, eq):
    """hole (N,6,2), board (N,5), nboard (N,), max_street (N,) -> eq (N,4,6,64) float32 (NaN where street not reached)."""
    N = hole.shape[0]
    for i in prange(N):
        tmp = np.empty((6, 64), np.float64)
        for s in range(max_street[i] + 1):
            _street_equity(hole[i], board[i], nboard[i], s, tmp, i + 1)
            for p in range(6):
                for m in range(64):
                    eq[i, s, p, m] = tmp[p, m]

@njit(cache=True)
def active_masks(hrow, seat, is_fold, n):
    """Active-seat bitmask BEFORE each action (actions sorted by hand, action_no); folds remove the actor after acting."""
    out = np.empty(n, np.int64)
    cur = 63; prev = -1
    for i in range(n):
        if hrow[i] != prev:
            cur = 63; prev = hrow[i]
        out[i] = cur
        if is_fold[i]:
            cur &= ~(1 << seat[i])
    return out

# ----------------------------------------------------------------------------------------------------------------------
# 3. data assembly
# ----------------------------------------------------------------------------------------------------------------------
def load_hands(hand_ids):
    hid = pl.Series("hand_id", list(hand_ids))
    hands = pl.scan_parquet(D + "hands.parquet").filter(pl.col("hand_id").is_in(hid.implode())).select(
        ["hand_id", "big_blind", "board_cards", "final_pot", "button_seat"]).collect().sort("hand_id").with_row_index("hrow")
    seats = pl.scan_parquet(D + "seats.parquet").filter(pl.col("hand_id").is_in(hid.implode())).select(
        ["hand_id", "player_id", "seat_no", "hole_card_1", "hole_card_2", "total_contribution", "net_chips", "folded", "went_to_showdown", "won_share"]).collect()
    seats = seats.join(hands.select(["hand_id", "hrow", "big_blind"]), on="hand_id").with_columns(
        pl.col("hole_card_1").replace_strict(CARD_MAP, return_dtype=pl.Int64).alias("h1"),
        pl.col("hole_card_2").replace_strict(CARD_MAP, return_dtype=pl.Int64).alias("h2"))
    assert seats.group_by("hand_id").agg(pl.col("seat_no").sort().alias("s")).filter(pl.col("s") != [0, 1, 2, 3, 4, 5]).height == 0, "seat_no must be 0..5"
    N = hands.height
    hole = np.zeros((N, 6, 2), np.int64)
    hole[seats["hrow"].to_numpy(), seats["seat_no"].to_numpy(), 0] = seats["h1"].to_numpy()
    hole[seats["hrow"].to_numpy(), seats["seat_no"].to_numpy(), 1] = seats["h2"].to_numpy()
    bl = hands["board_cards"].fill_null("").str.strip_chars().str.split(" ").list.eval(pl.element().replace_strict(CARD_MAP, default=-1, return_dtype=pl.Int64))
    board = np.column_stack([bl.list.get(i, null_on_oob=True).fill_null(-1).to_numpy() for i in range(5)]).astype(np.int64)
    nboard = (board >= 0).sum(axis=1).astype(np.int64)
    return hands, seats, hole, board, nboard

def equity_tables(hands, hole, board, nboard):
    max_street = np.where(nboard >= 5, 3, np.where(nboard >= 4, 2, np.where(nboard >= 3, 1, 0))).astype(np.int64)
    eq = np.full((hands.height, 4, 6, 64), np.nan, np.float32)
    equity_all(hole, board, nboard, max_street, eq)
    return eq

def action_frame(hands, seats, eq):
    hid = hands["hand_id"]
    ac = pl.scan_parquet(D + "actions.parquet").filter(pl.col("hand_id").is_in(hid.implode())).collect()
    ac = ac.join(seats.select(["hand_id", "player_id", "seat_no", "hrow", "big_blind"]), on=["hand_id", "player_id"]).sort(["hrow", "action_no"])
    ac = ac.with_columns(
        pl.col("street").replace_strict({"preflop": 0, "flop": 1, "turn": 2, "river": 3}, return_dtype=pl.Int8).alias("street_no"),
        (pl.col("action").is_in(["bet", "raise"]) | ((pl.col("action") == "all_in") & (pl.col("amount") > pl.col("to_call")))).alias("is_aggr"),
        (pl.col("action") == "fold").alias("is_fold"))
    ac = ac.with_columns(pl.when(pl.col("is_aggr")).then(pl.col("seat_no")).otherwise(None).alias("_ag")).with_columns(
        pl.col("_ag").shift(1).forward_fill().over(["hrow", "street_no"]).alias("last_aggr_seat"))
    mask = active_masks(ac["hrow"].to_numpy(), ac["seat_no"].to_numpy(), ac["is_fold"].to_numpy(), ac.height)
    ac = ac.with_columns(pl.Series("mask", mask))
    # sanity: reconstructed active count == players_active (recorded before the action)
    pc = np.array([bin(m).count("1") for m in range(64)])[mask]
    bad = (pc != ac["players_active"].to_numpy()).mean()
    log(f"active-mask mismatch vs players_active: {bad:.4%}")
    hr = ac["hrow"].to_numpy(); st = ac["street_no"].to_numpy().astype(np.int64); se = ac["seat_no"].to_numpy()
    ac = ac.with_columns(pl.Series("eq_all", eq[hr, st, se, mask]))
    return ac

# ----------------------------------------------------------------------------------------------------------------------
# 4. EV-transfer features per (pair_id, hand_id)
# ----------------------------------------------------------------------------------------------------------------------
def pair_hand_features(ac, pairs, hands, seats, eq):
    """pairs: (pair_id, hand_id, player_1, player_2). Returns one row per pair-hand."""
    sp = seats.select(["hand_id", "player_id", "seat_no"])
    pr = (pairs.join(sp.rename({"player_id": "player_1", "seat_no": "seat_1"}), on=["hand_id", "player_1"])
               .join(sp.rename({"player_id": "player_2", "seat_no": "seat_2"}), on=["hand_id", "player_2"]))
    # one row per (pair, hand, member): member seat + partner seat
    mem = pl.concat([
        pr.select(["pair_id", "hand_id", pl.col("seat_1").alias("seat_no"), pl.col("seat_2").alias("partner_seat"), pl.lit(1, dtype=pl.Int8).alias("member")]),
        pr.select(["pair_id", "hand_id", pl.col("seat_2").alias("seat_no"), pl.col("seat_1").alias("partner_seat"), pl.lit(2, dtype=pl.Int8).alias("member")])])
    a = ac.join(mem, on=["hand_id", "seat_no"])  # actions of pair members only (a hand shared by k pairs is duplicated k times)
    hr = a["hrow"].to_numpy(); st = a["street_no"].to_numpy().astype(np.int64); se = a["seat_no"].to_numpy(); ps = a["partner_seat"].to_numpy(); mk = a["mask"].to_numpy()
    partner_active = ((mk >> ps) & 1).astype(bool)
    hu_mask = (1 << se) | (1 << ps)
    eq_hu = np.where(partner_active, eq[hr, st, se, np.where(partner_active, hu_mask, mk)], np.nan)
    mk_np = mk & ~(1 << ps)
    has_other = (mk_np & ~(1 << se)) > 0
    eq_np = np.where(has_other, eq[hr, st, se, np.where(has_other, mk_np, mk)], np.nan)  # equity vs outsiders only
    a = a.with_columns(pl.Series("partner_active", partner_active), pl.Series("eq_hu", eq_hu), pl.Series("eq_np", eq_np),
                       pl.Series("n_active", np.array([bin(m).count("1") for m in range(64)])[mk]),
                       (pl.col("mask") == pl.Series(hu_mask)).alias("hu_partner"))
    bb = pl.col("big_blind")
    P = pl.col("pot_before") / bb; C = pl.col("to_call") / bb; Bt = pl.col("amount") / bb
    e = pl.col("eq_all"); eh = pl.col("eq_hu")
    facing_partner = (pl.col("to_call") > 0) & (pl.col("last_aggr_seat") == pl.col("partner_seat"))
    is_fold = pl.col("action") == "fold"; is_call = (pl.col("action") == "call") | ((pl.col("action") == "all_in") & ~pl.col("is_aggr"))
    is_check = pl.col("action") == "check"; is_bet = pl.col("is_aggr")
    ev_call = e * (P + C) - C                      # EV of calling under one-street pot odds (chips in bb)
    a = a.with_columns(
        facing_partner.alias("facing_partner"),
        # --- fold: value surrendered ---
        pl.when(is_fold).then(pl.max_horizontal(ev_call, 0.0)).otherwise(0.0).alias("fold_ev_lost"),
        pl.when(is_fold).then(e * P).otherwise(0.0).alias("fold_poteq"),
        pl.when(is_fold & pl.col("partner_active")).then(eh * P).otherwise(0.0).alias("fold_poteq_hu"),
        pl.when(is_fold & pl.col("partner_active")).then((eh > 0.5).cast(pl.Float32) * P).otherwise(0.0).alias("fold_ahead_pot"),
        # --- call: EV given away by a -EV call ---
        pl.when(is_call).then(pl.max_horizontal(-ev_call, 0.0)).otherwise(0.0).alias("call_ev_given"),
        pl.when(is_call & pl.col("partner_active")).then((1 - eh) * C).otherwise(0.0).alias("call_loss_hu"),
        # --- check: value surrendered by not value-betting a strong hand (pot-sized bet model) ---
        pl.when(is_check & pl.col("partner_active")).then(pl.max_horizontal(2 * e - 1, 0.0) * P).otherwise(0.0).alias("check_ev_surr"),
        pl.when(is_check & pl.col("partner_active")).then(pl.max_horizontal(2 * eh - 1, 0.0) * P).otherwise(0.0).alias("check_ev_surr_hu"),
        # --- bet/raise with a weak hand: value given (increment beyond the call is -EV when e<0.5) ---
        pl.when(is_bet & pl.col("partner_active")).then(pl.max_horizontal(1 - 2 * e, 0.0) * pl.max_horizontal(Bt - C, 0.0)).otherwise(0.0).alias("bet_ev_given"),
        pl.when(is_bet & pl.col("partner_active")).then(pl.max_horizontal(1 - 2 * eh, 0.0) * pl.max_horizontal(Bt - C, 0.0)).otherwise(0.0).alias("bet_ev_given_hu"),
        # generic one-street regret = best alternative - taken
        pl.when(pl.col("to_call") > 0).then(pl.max_horizontal(0.0, ev_call, e * (3 * P + 3 * C) - P - 2 * C))
          .otherwise(pl.max_horizontal(e * P, (3 * e - 1) * P)).alias("ev_best"),
        pl.when(is_fold).then(0.0).when(is_call).then(ev_call).when(is_check).then(e * P)
          .otherwise(e * (P + 2 * Bt - C) - Bt).alias("ev_taken"),
    ).with_columns(
        pl.max_horizontal(pl.col("ev_best") - pl.col("ev_taken"), 0.0).alias("regret"),
        (pl.col("fold_ev_lost") * facing_partner.cast(pl.Float32)).alias("fold_ev_to_partner"),
        (pl.col("fold_poteq") * facing_partner.cast(pl.Float32)).alias("fold_poteq_to_partner"),
        (pl.col("call_ev_given") * facing_partner.cast(pl.Float32)).alias("call_ev_to_partner"),
    ).with_columns(
        (pl.col("regret") * pl.col("partner_active").cast(pl.Float32)).alias("regret_pa"),
        (pl.col("regret") * pl.col("hu_partner").cast(pl.Float32)).alias("regret_hu"),
        (pl.col("fold_ev_to_partner") + pl.col("call_ev_to_partner") + pl.col("check_ev_surr") + pl.col("bet_ev_given")).alias("ev_transfer_act"),
        (pl.col("fold_poteq_hu") + pl.col("call_loss_hu") + pl.col("check_ev_surr_hu") + pl.col("bet_ev_given_hu")).alias("ev_transfer_hu_act"),
    )
    g = ["pair_id", "hand_id"]
    # per-street sums for the max-over-streets feature
    st_sum = a.group_by(g + ["street_no"]).agg(pl.col("ev_transfer_act").sum().alias("s"), pl.col("regret_pa").sum().alias("r"))
    st_max = st_sum.group_by(g).agg(pl.col("s").max().alias("evt_max_street"), pl.col("r").max().alias("regret_pa_max_street"),
                                    pl.col("s").arg_max().alias("_i"), pl.col("street_no").alias("_st")).with_columns(
        pl.col("_st").list.get(pl.col("_i")).alias("evt_argmax_street")).drop(["_i", "_st"])
    # per-member sums -> directionality
    mem_sum = a.group_by(g + ["member"]).agg(pl.col("ev_transfer_act").sum().alias("t"), pl.col("regret_pa").sum().alias("r"))
    mem_w = mem_sum.pivot(on="member", index=g, values=["t", "r"]).fill_null(0.0)
    for c in ["t_1", "t_2", "r_1", "r_2"]:
        if c not in mem_w.columns:
            mem_w = mem_w.with_columns(pl.lit(0.0).alias(c))
    mem_w = mem_w.select(g + [pl.max_horizontal("t_1", "t_2").alias("evt_giver_max"), pl.min_horizontal("t_1", "t_2").alias("evt_giver_min"),
                             (pl.col("t_1") - pl.col("t_2")).abs().alias("evt_asym"), pl.max_horizontal("r_1", "r_2").alias("regret_pa_member_max")])
    feats = a.group_by(g).agg(
        pl.col("fold_ev_lost").max().alias("fold_ev_lost_max"),
        pl.col("fold_ev_to_partner").max().alias("fold_ev_to_partner_max"),
        pl.col("fold_poteq_to_partner").max().alias("fold_poteq_to_partner_max"),
        pl.col("fold_poteq_hu").max().alias("fold_poteq_hu_max"),
        pl.col("fold_ahead_pot").max().alias("fold_ahead_pot_max"),
        pl.col("call_ev_given").max().alias("call_ev_given_max"),
        pl.col("call_ev_to_partner").max().alias("call_ev_to_partner_max"),
        pl.col("call_ev_to_partner").sum().alias("call_ev_to_partner_sum"),
        pl.col("call_loss_hu").sum().alias("call_loss_hu_sum"),
        pl.col("check_ev_surr").max().alias("check_ev_surr_max"),
        pl.col("check_ev_surr").sum().alias("check_ev_surr_sum"),
        pl.col("check_ev_surr_hu").max().alias("check_ev_surr_hu_max"),
        pl.col("bet_ev_given").max().alias("bet_ev_given_max"),
        pl.col("bet_ev_given_hu").sum().alias("bet_ev_given_hu_sum"),
        pl.col("regret").sum().alias("regret_sum"),
        pl.col("regret").max().alias("regret_max"),
        pl.col("regret_pa").sum().alias("regret_pa_sum"),
        pl.col("regret_pa").max().alias("regret_pa_max"),
        pl.col("regret_hu").sum().alias("regret_hu_sum"),
        pl.col("ev_transfer_act").sum().alias("ev_transfer"),
        pl.col("ev_transfer_act").max().alias("ev_transfer_actmax"),
        pl.col("ev_transfer_hu_act").sum().alias("ev_transfer_hu"),
        # equity context of the pair's decisions
        pl.col("eq_hu").filter(pl.col("action").is_in(["call", "fold"]) & pl.col("facing_partner")).min().alias("eq_hu_min_facing_partner"),
        pl.col("eq_hu").filter(pl.col("is_aggr")).min().alias("eq_hu_min_when_aggr"),
        pl.col("eq_hu").filter(pl.col("action") == "check").max().alias("eq_hu_max_when_check"),
        pl.col("eq_all").filter(pl.col("street_no") == 0).min().alias("eq_pf_min"),
        pl.col("eq_all").filter((pl.col("street_no") == 0) & ~pl.col("is_fold")).min().alias("eq_pf_min_entered"),
        (pl.col("eq_all").filter((pl.col("street_no") == 0) & ~pl.col("is_fold")) < 0.12).sum().alias("n_pf_enter_lowe"),
        pl.col("eq_np").filter(pl.col("street_no") > 0).max().alias("eq_vs_outsiders_max"),
        pl.col("partner_active").filter(pl.col("street_no") > 0).any().cast(pl.Float32).alias("both_alive_postflop"),
        pl.col("hu_partner").any().cast(pl.Float32).alias("hu_partner_any"),
    ).join(st_max, on=g, how="left").join(mem_w, on=g, how="left")
    # ---- showdown / final-street expected-value transfer between the partners ----
    hs = hands.select(["hand_id", "hrow", "big_blind", "final_pot"])
    sd = pr.join(hs, on="hand_id")
    s1 = seats.select(["hand_id", "seat_no", "net_chips", "total_contribution", "went_to_showdown", "folded"])
    sd = (sd.join(s1.rename({"seat_no": "seat_1", "net_chips": "net_1", "total_contribution": "con_1", "went_to_showdown": "sd_1", "folded": "fold_1"}), on=["hand_id", "seat_1"])
            .join(s1.rename({"seat_no": "seat_2", "net_chips": "net_2", "total_contribution": "con_2", "went_to_showdown": "sd_2", "folded": "fold_2"}), on=["hand_id", "seat_2"]))
    # last street with both partners un-folded: equity of each vs the full active set at that street start (masks from action_frame)
    # approximate via the final street reached by the hand (max_street) and the seats not folded at that point = seats with went_to_showdown or hand ended earlier
    hr = sd["hrow"].to_numpy()
    last_st = (np.isfinite(eq[hr, :, 0, 63])).sum(axis=1) - 1  # last street the hand reached (0..3)
    s1_ = sd["seat_1"].to_numpy(); s2_ = sd["seat_2"].to_numpy()
    e1_last = eq[hr, last_st, s1_, (1 << s1_) | (1 << s2_)]; e1_pf = eq[hr, 0, s1_, (1 << s1_) | (1 << s2_)]
    # equity of loser (lower net) vs partner at the last street and at preflop, times final pot
    net1 = sd["net_1"].to_numpy(); net2 = sd["net_2"].to_numpy(); bbv = sd["big_blind"].to_numpy(); pot = sd["final_pot"].to_numpy() / bbv
    loser_is_1 = net1 < net2
    e_loser_last = np.where(loser_is_1, e1_last, 1 - e1_last); e_loser_pf = np.where(loser_is_1, e1_pf, 1 - e1_pf)
    con_l = np.where(loser_is_1, sd["con_1"].to_numpy(), sd["con_2"].to_numpy()) / bbv
    net_l = np.where(loser_is_1, net1, net2) / bbv
    sd = sd.with_columns(
        pl.Series("sd_eq_loser_last", e_loser_last.astype(np.float32)),
        pl.Series("sd_eq_loser_pf", e_loser_pf.astype(np.float32)),
        pl.Series("sd_ev_loser_last_pot", (e_loser_last * pot).astype(np.float32)),
        pl.Series("sd_ev_expect_loser", (e_loser_last * pot - con_l).astype(np.float32)),
        pl.Series("sd_luck_loser", (net_l - (e_loser_last * pot - con_l)).astype(np.float32)),  # actual - expected (negative = unlucky)
        pl.Series("pair_pot_bb", pot.astype(np.float32)),
    ).select(["pair_id", "hand_id", "sd_eq_loser_last", "sd_eq_loser_pf", "sd_ev_loser_last_pot", "sd_ev_expect_loser", "sd_luck_loser", "pair_pot_bb"])
    out = pairs.select(["pair_id", "hand_id"]).join(feats, on=g, how="left").join(sd, on=g, how="left")
    fill0 = [c for c in feats.columns if c not in g and c not in ("eq_hu_min_facing_partner", "eq_hu_min_when_aggr", "eq_hu_max_when_check", "eq_pf_min", "eq_pf_min_entered", "eq_vs_outsiders_max", "evt_argmax_street")]
    out = out.with_columns([pl.col(c).fill_null(0.0) for c in fill0])
    out = out.with_columns((pl.col("ev_transfer") / pl.max_horizontal(pl.col("pair_pot_bb"), 1.0)).alias("ev_transfer_frac_pot"),
                           (pl.col("regret_pa_sum") / pl.max_horizontal(pl.col("pair_pot_bb"), 1.0)).alias("regret_pa_frac_pot"))
    return out

PRANK_COLS = ["ev_transfer", "ev_transfer_hu", "evt_max_street", "regret_pa_sum", "regret_sum", "fold_ev_to_partner_max", "fold_poteq_to_partner_max",
              "check_ev_surr_max", "call_ev_to_partner_max", "bet_ev_given_max", "ev_transfer_frac_pot", "sd_ev_loser_last_pot", "sd_luck_loser", "evt_giver_max"]

def add_pranks(out):
    return out.with_columns([(pl.col(c).rank(method="average").over("pair_id") / pl.len().over("pair_id")).cast(pl.Float32).alias(c + "_prank") for c in PRANK_COLS])

def build_features(pairs):
    hand_ids = pairs["hand_id"].unique()
    log(f"{pairs.height} pair-hands, {len(hand_ids)} unique hands")
    hands, seats, hole, board, nboard = load_hands(hand_ids)
    t = time.time(); eq = equity_tables(hands, hole, board, nboard); log(f"equity tables {eq.shape} in {time.time()-t:.0f}s")
    ac = action_frame(hands, seats, eq)
    out = pair_hand_features(ac, pairs, hands, seats, eq)
    out = add_pranks(out)
    return out, eq, hands, seats, ac

# ----------------------------------------------------------------------------------------------------------------------
# 5. evaluation (dev only)
# ----------------------------------------------------------------------------------------------------------------------
def map5(df, score_col):
    """host formula: per positive pair sort by score desc (ties pot_bb desc, hand_id asc); AP@5 over labelled set / min(n_rel,5)."""
    d = df.sort(["pair_id", score_col, "pot_bb", "hand_id"], descending=[False, True, True, False])
    aps = []
    for pid, g in d.group_by("pair_id", maintain_order=True):
        rel = g["is_ev"].to_numpy(); n_rel = rel.sum()
        if n_rel == 0:
            continue
        top = rel[:5]; hits = 0; s = 0.0
        for k in range(len(top)):
            if top[k]:
                hits += 1; s += hits / (k + 1)
        aps.append(s / min(n_rel, 5))
    return float(np.mean(aps))

def harness(df, feats, seeds=(42, 49), rounds=400):
    import lightgbm as lgb
    X = df.select(feats).to_numpy().astype(np.float32); y = df["is_ev"].to_numpy().astype(int); fold = df["fold"].to_numpy()
    P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=3)
    oof = np.zeros(len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        for sd in seeds:
            m = lgb.train({**P, "seed": sd}, lgb.Dataset(X[tr], y[tr]), num_boost_round=rounds); oof[te] += m.predict(X[te]) / len(seeds)
    return oof

def evaluate(out):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    lab = pl.read_csv(D + "development_labels.csv"); ev = pl.read_csv(D + "development_evidence.csv")
    pos = lab.filter(pl.col("label") == 1); ids = set(pos["pair_id"])
    src = open(f"{ROOT}/step9_pipeline.py").read(); ns = {}; exec(src[src.index("HAND_RAW = ["):src.index("DEV_HF = [")], ns); HF = ns["HAND_FEATS"]
    cols = ["pair_id", "hand_id", "table_id", "hand_idx", "pot_bb", "surp_pair_max"] + [c for c in HF if c not in ("pot_bb", "surp_pair_max")]
    df = pl.concat([pl.scan_parquet(f"{W}dev_hand_features_v9_{i}.parquet").filter(pl.col("pair_id").is_in(list(ids))).select(cols).collect() for i in range(4)])
    df = df.join(ev.select(["pair_id", "hand_id", pl.lit(True).alias("is_ev"), "evidence_rank"]), on=["pair_id", "hand_id"], how="left").with_columns(pl.col("is_ev").fill_null(False)).join(pos.select(["pair_id", "behavior_family"]), on="pair_id")
    tables = sorted(df["table_id"].unique().to_list()); rng = np.random.RandomState(42); TF = {t: int(v) for t, v in zip(tables, rng.permutation(len(tables)) % 5)}
    df = df.with_columns(pl.col("table_id").replace_strict(TF, return_dtype=pl.Int8).alias("fold")).sort(["pair_id", "hand_idx"])
    for f in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        df = df.with_columns((pl.col("behavior_family") == f).cast(pl.Float32).alias("fam_" + f))
    df = df.join(out, on=["pair_id", "hand_id"], how="left")
    NEW = [c for c in out.columns if c not in ("pair_id", "hand_id")]
    log(f"harness frame {df.shape}; new features: {len(NEW)}; null rate of ev_transfer {df['ev_transfer'].null_count()/df.height:.4f}")
    y = df["is_ev"].to_numpy(); conf = (~df["is_ev"]).to_numpy() & (df["surp_pair_max"].to_numpy() >= 3)
    auc_all, auc_conf = {}, {}
    for c in NEW:
        x = df[c].to_numpy().astype(np.float64); ok = np.isfinite(x)
        if ok.sum() < 100 or y[ok].sum() == 0 or (~y[ok]).sum() == 0:
            continue
        auc_all[c] = float(roc_auc_score(y[ok], x[ok]))
        m2 = ok & (y | conf)
        auc_conf[c] = float(roc_auc_score(y[m2], x[m2]))
    log("univariate AUC (is_ev vs all | is_ev vs confusers):")
    for c in sorted(auc_all, key=lambda c: -abs(auc_all[c] - 0.5)):
        print(f"  {c:32s} {auc_all[c]:.3f}  {auc_conf[c]:.3f}")
    # diagnostic: within-pair spearman of evidence_rank vs feature among the labelled hands (label used only here)
    evd = df.filter(pl.col("is_ev"))
    diag = {}
    for c in ["ev_transfer", "ev_transfer_hu", "regret_pa_sum", "fold_ev_to_partner_max", "check_ev_surr_max", "sd_ev_loser_last_pot", "pair_pot_bb"]:
        rs = []
        for pid, g in evd.group_by("pair_id"):
            if g.height >= 4 and g[c].n_unique() > 1:
                r = spearmanr(g["evidence_rank"].to_numpy(), g[c].to_numpy()).correlation
                if np.isfinite(r):
                    rs.append(r)
        diag[c] = (float(np.mean(rs)), len(rs))
    log("diagnostic mean spearman(evidence_rank, feature) within pair over labelled hands (negative = larger value earlier rank):", diag)
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        sub = evd.filter(pl.col("behavior_family") == fam); rs = []
        for pid, g in sub.group_by("pair_id"):
            if g.height >= 4 and g["ev_transfer"].n_unique() > 1:
                r = spearmanr(g["evidence_rank"].to_numpy(), g["ev_transfer"].to_numpy()).correlation
                if np.isfinite(r): rs.append(r)
        print(f"  ev_transfer spearman by family {fam}: {np.mean(rs):.3f} (n={len(rs)})")
    # MAP@5 harness
    FE0 = HF + ["fam_directed_transfer", "fam_soft_play", "fam_coordinated_isolation"]
    FE1 = FE0 + NEW
    t = time.time(); oof0 = harness(df, FE0); df = df.with_columns(pl.Series("s0", oof0)); m0 = map5(df, "s0"); log(f"baseline MAP@5 {m0:.4f} ({time.time()-t:.0f}s)")
    t = time.time(); oof1 = harness(df, FE1); df = df.with_columns(pl.Series("s1", oof1)); m1 = map5(df, "s1"); log(f"with ev_transfer MAP@5 {m1:.4f} ({time.time()-t:.0f}s)")
    # per-family and per-fold breakdown
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        sub = df.filter(pl.col("behavior_family") == fam); print(f"  {fam:24s} base {map5(sub,'s0'):.4f} with {map5(sub,'s1'):.4f}")
    for f in range(5):
        sub = df.filter(pl.col("fold") == f); print(f"  fold {f} base {map5(sub,'s0'):.4f} with {map5(sub,'s1'):.4f}")
    # AUC of the OOF scores vs confusers
    from sklearn.metrics import roc_auc_score as ras
    m2 = y | conf
    log(f"OOF AUC ev vs confusers: base {ras(y[m2], oof0[m2]):.3f} with {ras(y[m2], oof1[m2]):.3f}; ev vs all: base {ras(y, oof0):.3f} with {ras(y, oof1):.3f}")
    df.select(["pair_id", "hand_id", "is_ev", "s0", "s1"]).write_parquet(H + "ev_transfer_oof.parquet")
    return dict(map5_baseline=m0, map5_with=m1, auc_all=auc_all, auc_conf=auc_conf, diag=diag)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="dev"); ap.add_argument("--pairs", default=None); ap.add_argument("--out", default=None); ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    if args.pairs:
        pairs = pl.read_parquet(args.pairs).select(["pair_id", "hand_id", "player_1", "player_2"])
    elif args.phase == "dev":
        lab = pl.read_csv(D + "development_labels.csv").filter(pl.col("label") == 1).select(["pair_id", "player_1", "player_2"])
        pairs = pl.read_parquet(W + "pos_hand_oof_v9.parquet").select(["pair_id", "hand_id"]).join(lab, on="pair_id")
    else:
        pairs = pl.read_parquet(W + "eval_pair_hands.parquet").select(["pair_id", "hand_id", "player_1", "player_2"])
    out, eq, hands, seats, ac = build_features(pairs)
    dst = args.out or (H + "ev_transfer.parquet" if args.phase == "dev" else H + "ev_transfer_eval.parquet")
    out.write_parquet(dst); log(f"saved {dst} {out.shape}")
    if args.phase == "dev" and not args.no_eval:
        res = evaluate(out)
        import json; json.dump(res, open(H + "ev_transfer_result.json", "w"), indent=1); log("done")
