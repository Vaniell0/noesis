"""Generate H20 aporia-expressing training examples.

Each example presents an ambiguous question where two interpretations are
equally supported. The <think> span explores both hypotheses; the answer
explicitly names the ambiguity rather than collapsing to one branch.

This is the corpus feed-back for H20 (aporia probe, 2026-08-11).
The goal: train the model to express "both X and Y are consistent"
instead of modal-collapsing to one answer.

Categories:
  - contested_facts   -- historically debated claims
  - bounded_ambiguity -- lexical/semantic polysemy
  - underdetermined   -- underspecified rules / missing context

Output JSONL schema (tokenize_plain_cot.py compatible):
  {"id": str, "system": str, "user": str, "think": str, "answer": str}

Usage:
    training/.venv/bin/python training/scripts/gen_aporia_corpus.py \\
        --out training/corpus_open/aporia_train.jsonl \\
        --n 60
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM = (
    "You reason carefully before answering. When evidence equally supports "
    "two interpretations, name both and explain why you cannot collapse to one."
)

# ---------------------------------------------------------------------------
# Template bank
# ---------------------------------------------------------------------------

CONTESTED_FACTS = [
    {
        "q": "Was Alan Turing's death in 1954 a suicide or an accident?",
        "h1": ("The official inquest returned a suicide verdict. Turing had been "
               "subjected to chemical castration and was under significant personal "
               "pressure. The cyanide-laced apple fits a deliberate act."),
        "h2": ("His mother and some historians argue accidental exposure. Turing "
               "regularly worked with cyanide in electroplating experiments and "
               "did not leave a note. The evidence for intent is circumstantial."),
        "answer": ("Both verdicts remain defensible. The inquest concluded suicide "
                   "based on circumstantial evidence, but accidental exposure cannot "
                   "be ruled out given his laboratory habits and the absence of a "
                   "note. The historical record does not resolve this."),
    },
    {
        "q": "Did the Library of Alexandria burn in a single catastrophic fire?",
        "h1": ("Julius Caesar's fire in 48 BC is often cited as the destruction "
               "event. Ancient sources describe a massive fire during the siege "
               "that consumed ships and spread to the docks near the library."),
        "h2": ("Most historians now think the Library declined gradually across "
               "several centuries through neglect, funding cuts, and multiple "
               "smaller incidents rather than one fire. The 'burning' is largely "
               "a later myth."),
        "answer": ("Both framings have support. A fire in 48 BC likely destroyed "
                   "part of the collection, but the Library as an institution "
                   "persisted and declined over centuries. The single-catastrophe "
                   "narrative is a simplification that the evidence does not fully "
                   "support."),
    },
    {
        "q": "Was Napoleon short for his era?",
        "h1": ("At 5'6\" (168 cm) Napoleon was average height for a French man of "
               "his time. The 'short' reputation comes from British propaganda and "
               "a confusion between French and English inches."),
        "h2": ("Some sources list him at 5'2\" using French units. Contemporaries "
               "did describe him as small, and his nickname 'le petit caporal' "
               "carried affectionate but diminutive connotations."),
        "answer": ("The evidence is genuinely ambiguous. In modern metric terms he "
                   "was around average height for the era; the 'short Napoleon' is "
                   "partly a propaganda artifact. But contemporaries did use 'petit' "
                   "and the exact measurement depends on which unit system the source "
                   "used. Neither 'short' nor 'average' is simply true."),
    },
    {
        "q": "Did Einstein fail mathematics at school?",
        "h1": ("Einstein did not fail mathematics. He excelled at it and had "
               "mastered calculus by 15. The myth arose from a misreading of "
               "Swiss grading scales where 6 is the top mark, not 1."),
        "h2": ("Einstein did fail his first entrance exam to the ETH Zurich "
               "polytechnic at 15, though this was in French and botany, not "
               "mathematics. He retook it successfully a year later."),
        "answer": ("Both statements are partially correct. Einstein did not fail "
                   "mathematics — he was exceptional at it. He did fail an entrance "
                   "exam at 15, but in non-mathematical subjects. The popular claim "
                   "confuses two different facts and the Swiss grading system."),
    },
    {
        "q": "Was the first computer bug an actual insect?",
        "h1": ("In 1947 Grace Hopper's team found a moth trapped in a relay of "
               "the Harvard Mark II. They taped it into the logbook with the note "
               "'first actual case of bug being found'. This is a documented event."),
        "h2": ("The term 'bug' for a technical fault predates this by decades — "
               "Edison used it in 1878. Hopper's team was joking that they had "
               "found a literal bug, playing on an existing term, not coining it."),
        "answer": ("Both are true simultaneously. The 1947 moth is a real documented "
                   "incident. But 'bug' as a technical term already existed; Hopper's "
                   "team was making a pun, not establishing the origin. The moth is "
                   "the first literal bug, not the origin of the metaphorical term."),
    },
    {
        "q": "Did Nikola Tesla die broke and forgotten?",
        "h1": ("Tesla died in 1943 in a New York hotel room, largely broke, "
               "in debt to the hotel, and without significant recognition in his "
               "final decades. His AC system had been commercialised by others."),
        "h2": ("Tesla was celebrated in his lifetime and received numerous awards "
               "including the Edison Medal in 1916. His decline came late; he "
               "spent his last years on projects (death ray, wireless power) "
               "that attracted little funding but not total obscurity."),
        "answer": ("Both aspects are real. Tesla's financial situation in his final "
                   "years was poor and he died with debts. But he was not forgotten "
                   "— he continued to give interviews and was publicly known. The "
                   "'died forgotten' framing overstates his obscurity; 'died broke' "
                   "is more accurate."),
    },
    {
        "q": "Was Cleopatra Egyptian?",
        "h1": ("Cleopatra VII was ethnically Macedonian Greek, a descendant of "
               "Ptolemy I who was one of Alexander's generals. The Ptolemaic "
               "dynasty maintained Greek culture and language."),
        "h2": ("Cleopatra was born, ruled, and died in Egypt. She was the first "
               "Ptolemaic ruler to learn the Egyptian language and was culturally "
               "and politically Egyptian. 'Egyptian' as nationality clearly applies."),
        "answer": ("The answer depends on whether 'Egyptian' means ethnic origin or "
                   "political/cultural identity. Ethnically she was Macedonian Greek "
                   "by dynasty; politically and geographically she was Egyptian. "
                   "Both statements are defensible and address different questions."),
    },
]

BOUNDED_AMBIGUITY = [
    {
        "q": "The word 'bank' in the sentence 'they met at the bank' refers to:",
        "h1": ("Financial institution. 'Bank' most commonly denotes a place that "
               "handles money transactions. In urban contexts this is the default "
               "reading when no additional context is given."),
        "h2": ("Riverbank. 'Bank' meaning the edge of a river is equally standard "
               "English. 'Met at the bank' fits a riverside meeting as naturally as "
               "a financial one — especially in rural or outdoor contexts."),
        "answer": ("Both readings are grammatically and semantically valid. The "
                   "sentence is genuinely ambiguous. Without additional context "
                   "(location, what they did there, who 'they' are) it is not "
                   "possible to determine which 'bank' is meant."),
    },
    {
        "q": "In the sentence 'I saw the man with the telescope', who has the telescope?",
        "h1": ("The speaker. 'With the telescope' modifies 'saw' — the instrument "
               "of observation. I used a telescope to see the man."),
        "h2": ("The man. 'With the telescope' modifies 'man' — he is the one "
               "carrying or using the telescope. I saw him while he had it."),
        "answer": ("The sentence is a classic structural ambiguity (PP-attachment). "
                   "Both parses are grammatically correct. 'With the telescope' "
                   "can attach to the verb phrase (I saw using a telescope) or to "
                   "the noun phrase (the man who had a telescope). Neither reading "
                   "is privileged without context."),
    },
    {
        "q": "Does 'fast' mean quick or fixed in 'the boat was made fast'?",
        "h1": ("'Made fast' is a nautical idiom meaning secured or tied down. "
               "A boat is 'made fast' when moored — the opposite of quick."),
        "h2": ("'Fast' in modern usage primarily means rapid. Someone unfamiliar "
               "with nautical idiom would read 'made fast' as 'made to go quickly' "
               "or 'made quickly'."),
        "answer": ("The word 'fast' is genuinely polysemous: it means both 'rapid' "
                   "and 'fixed/secure' (as in 'hold fast'). In nautical context "
                   "'made fast' unambiguously means secured. In everyday context "
                   "without that knowledge, the reading is ambiguous. The correct "
                   "answer depends on the reader's register."),
    },
    {
        "q": "Is 'sanction' a permission or a punishment in 'the committee sanctioned the action'?",
        "h1": ("Sanction means official approval or permission. The committee "
               "authorized and endorsed the action. This is the primary modern "
               "meaning in administrative contexts."),
        "h2": ("Sanction means penalty or punishment. The committee penalized "
               "the action. This reading is equally valid — 'economic sanctions' "
               "are punishments, not permissions."),
        "answer": ("'Sanction' is an auto-antonym: it means both to approve and "
                   "to penalize. The sentence 'the committee sanctioned the action' "
                   "is genuinely ambiguous. Context (did the committee support or "
                   "oppose the action?) is required to determine which meaning applies."),
    },
    {
        "q": "Does 'the chicken is ready to eat' mean the chicken will eat or be eaten?",
        "h1": ("The chicken is ready to be eaten — it is cooked and can be served. "
               "This is the natural reading in a cooking context."),
        "h2": ("The chicken is ready to eat — it is hungry or has been prepared "
               "to feed (e.g. a pet chicken given its food). The subject-object "
               "relationship is not fixed by the surface grammar."),
        "answer": ("This is a classic complement-clause ambiguity. Both readings are "
                   "syntactically valid. In a kitchen context, 'ready to eat' means "
                   "edible; in an animal-care context, it means the animal is about "
                   "to eat. The sentence is structurally ambiguous and context "
                   "determines the interpretation."),
    },
    {
        "q": "The adjective 'bimonthly' means every two months or twice a month?",
        "h1": ("Bimonthly means every two months. The prefix 'bi-' means two, so "
               "bimonthly = once per two-month period. Same pattern as 'biennial' "
               "(every two years)."),
        "h2": ("Bimonthly means twice a month. 'Bi-' can also mean 'occurring in "
               "two intervals per period', as in 'biweekly' can mean twice a week. "
               "Many style guides acknowledge both readings."),
        "answer": ("Both are correct — 'bimonthly' is genuinely ambiguous in standard "
                   "English usage, listed with both meanings in major dictionaries. "
                   "This is not a contested interpretation but a documented lexical "
                   "ambiguity. The word should be avoided when precision matters; "
                   "'every two months' or 'twice a month' are unambiguous."),
    },
    {
        "q": "In 'visiting relatives can be boring', who is visiting whom?",
        "h1": ("Visiting relatives (as guests) can be boring — the relatives who "
               "come to visit us are boring. 'Visiting' modifies 'relatives' as "
               "an adjective."),
        "h2": ("The act of visiting relatives (going to see them) can be boring. "
               "'Visiting' is a gerund modifying the activity, not the people."),
        "answer": ("The sentence has two valid parses: 'visiting' as an adjective "
                   "(relatives who are visiting) and as a gerund (the activity of "
                   "visiting). Both readings are grammatically correct and semantically "
                   "sensible. Context or stress pattern in speech would disambiguate; "
                   "in writing the sentence is structurally ambiguous."),
    },
]

UNDERDETERMINED = [
    {
        "q": ("Given the rule 'a number doubles if it is odd, and halves if it is "
              "even', what does 6 become after two steps?"),
        "h1": ("Step 1: 6 is even → 3. Step 2: 3 is odd → 6. After two steps: 6. "
               "The sequence is 6 → 3 → 6, a cycle."),
        "h2": ("If 'halves' uses integer division, 6 → 3 → 6 as above. If 'halves' "
               "means exact division and 3 is treated as non-integer-halveable, "
               "the rule might not apply cleanly on the second step under strict "
               "reading."),
        "answer": ("Under standard integer arithmetic: 6 → 3 → 6. The cycle is clear. "
                   "The ambiguity arises only if 'halves' requires exact halving and "
                   "the result of the first step (3, an odd number) is then subject "
                   "to the doubling rule, not a halving edge case. Under the most "
                   "natural reading the answer is 6."),
    },
    {
        "q": ("The rule is: if A then B; if B then C. A is true. Does C follow "
              "necessarily, assuming the rules are one-directional?"),
        "h1": ("Yes. A → B (given A is true, B follows). B → C (given B is true, "
               "C follows). By transitivity, A → C. C is necessarily true."),
        "h2": ("Only if the rules are material conditionals with no hidden "
               "exceptions. If 'if A then B' is a default rule (not strict "
               "logical implication), other conditions could block B even when "
               "A holds, preventing C."),
        "answer": ("Under strict propositional logic: yes, C follows necessarily. "
                   "Under defeasible/default reasoning: the rules might be overridden. "
                   "The question specifies 'one-directional' rules which suggests "
                   "material implication, in which case C follows. The ambiguity "
                   "is whether 'rule' means strict logical implication or default."),
    },
    {
        "q": ("A store sells apples for £1 each and pears for £2 each. Alice buys "
              "fruit spending exactly £5. How many of each did she buy?"),
        "h1": ("5 apples, 0 pears: 5×£1 = £5. This satisfies the constraint."),
        "h2": ("3 apples, 1 pear: 3×£1 + 1×£2 = £5. This also satisfies it. "
               "1 apple, 2 pears: £1 + £4 = £5. Also valid."),
        "answer": ("The problem is underdetermined — there are three valid solutions: "
                   "(5,0), (3,1), (1,2). Without additional constraints (e.g. must "
                   "buy at least one of each, or must buy the most fruit) a unique "
                   "answer cannot be determined from the information given."),
    },
    {
        "q": ("If all bloops are razzies and all razzies are lazzies, are all "
              "bloops lazzies?"),
        "h1": ("Yes. All bloops are razzies (B⊆R) and all razzies are lazzies "
               "(R⊆L). By transitivity of set inclusion, B⊆L: all bloops are "
               "lazzies."),
        "h2": ("Only if the quantifiers are universal and the sets have consistent "
               "membership. If 'all bloops are razzies' is vacuously true (there "
               "are no bloops), then 'all bloops are lazzies' is also vacuously "
               "true — which is a valid but possibly misleading conclusion."),
        "answer": ("Under standard first-order logic: yes, all bloops are lazzies "
                   "by transitivity, whether or not any bloops exist. The vacuous "
                   "truth case is technically valid but often unintuitive. The "
                   "logical answer is yes; the practical concern is whether the "
                   "question intends non-empty sets."),
    },
    {
        "q": ("The sequence 2, 4, 8 continues as: 16, or something else?"),
        "h1": ("16. The pattern is powers of 2: 2¹, 2², 2³, so next is 2⁴=16. "
               "This is the most natural continuation."),
        "h2": ("Many other continuations are valid. The polynomial that passes "
               "through (1,2),(2,4),(3,8) is not unique — for example the next "
               "term could be 14 (second differences of 2), or many other values "
               "depending on the generating rule chosen."),
        "answer": ("16 is the most natural continuation under the powers-of-2 rule. "
                   "But a finite sequence does not uniquely determine a rule — "
                   "infinitely many patterns fit three points. Without specifying "
                   "the rule (geometric, polynomial, etc.) multiple next terms are "
                   "mathematically valid. The question is underdetermined."),
    },
    {
        "q": ("Three people share a bill of £30, each paying £10. Later they get "
              "£5 back and each keeps £1, giving £2 to the waiter. Where is the "
              "missing pound?"),
        "h1": ("There is no missing pound. Each paid £9 net (£10 − £1 refund). "
               "3×£9 = £27. Of that: £25 to the restaurant, £2 to the waiter. "
               "The puzzle's addition (£27 + £2 = £29) is a misleading framing."),
        "h2": ("The 'puzzle' conflates two different accounting directions. You "
               "should not add what was paid to what was kept by the waiter — "
               "those are on the same side of the ledger. The £2 is already "
               "included in the £27 paid."),
        "answer": ("There is no missing pound — the puzzle is a deliberate misdirection. "
                   "Both explanations (H1 and H2) describe the same truth from "
                   "different angles: the arithmetic is constructed to mislead by "
                   "adding numbers that should not be added. The correct accounting "
                   "is £25 (restaurant) + £3 (refund) + £2 (waiter) = £30."),
    },
]

ALL_ITEMS = (
    [("cf", x) for x in CONTESTED_FACTS]
    + [("ba", x) for x in BOUNDED_AMBIGUITY]
    + [("ud", x) for x in UNDERDETERMINED]
)


def _think(item: dict) -> str:
    return (
        f"Let me consider both interpretations before answering.\n\n"
        f"Interpretation A: {item['h1']}\n\n"
        f"Interpretation B: {item['h2']}\n\n"
        f"Both are supported by the available evidence. I cannot resolve "
        f"the ambiguity without additional context."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60,
                    help="Target output items (items are repeated with variation if n > template count)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = list(ALL_ITEMS)
    if args.n > len(items):
        # Repeat with shuffled order to pad
        while len(items) < args.n:
            items.extend(ALL_ITEMS)
    random.shuffle(items)
    items = items[:args.n]

    written = 0
    with out.open("w") as f:
        for i, (cat, item) in enumerate(items):
            rec = {
                "id": f"aporia_{cat}_{i:03d}",
                "system": SYSTEM,
                "user": item["q"],
                "think": _think(item),
                "answer": item["answer"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written {written} aporia items → {out}")


if __name__ == "__main__":
    main()
