"""Generate H21 invalid-premise refusal training examples.

Each example contains a question with a flawed premise. The <think> span
identifies the flaw before answering; the answer names it explicitly and
either corrects, qualifies, or refuses to answer as if the premise were true.

This is the corpus feed-back for H21 (premise-validity probe, 2026-08-11).
Goal: train the model to detect invalid premises in WKV state before output.

Premise invalidity types:
  impossible      -- logical contradiction or physical impossibility
  categorical     -- category error (asking attribute of wrong type)
  selfcontradiction -- self-referential contradiction
  counterfactual  -- false presupposition of fact
  arithmetic      -- false numerical assertion

Output JSONL schema (tokenize_plain_cot.py compatible):
  {"id": str, "system": str, "user": str, "think": str, "answer": str}

Usage:
    training/.venv/bin/python training/scripts/gen_premise_refusal_corpus.py \\
        --out training/corpus_open/premise_refusal_train.jsonl \\
        --n 60
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM = (
    "You check the validity of premises before reasoning. If a premise is "
    "false or contradictory, name the flaw first, then answer what can be "
    "answered. Do not reason from false premises as if they were true."
)

ITEMS = [
    # --- impossible ---
    {
        "type": "impossible",
        "q": "If you divide 7 by 0, what do you get?",
        "flaw": "Division by zero is undefined. There is no real number equal to 7÷0.",
        "think": ("The premise assumes 7÷0 has a value, but division by zero is "
                  "undefined in standard arithmetic. The expression has no result "
                  "in the real numbers. I should name this rather than produce a "
                  "number."),
        "answer": ("7÷0 is undefined — division by zero has no value in standard "
                   "arithmetic. No real number satisfies x·0 = 7."),
    },
    {
        "type": "impossible",
        "q": "What colour is the sound of middle C?",
        "flaw": "Sound has no colour. Colour is a visual property; sound is auditory.",
        "think": ("The question asks for a visual attribute (colour) of an auditory "
                  "phenomenon (a musical note). These are different sensory modalities. "
                  "Unless synesthesia is the topic, the premise contains a category error."),
        "answer": ("Sound does not have a colour in the literal sense — colour is a "
                   "visual property and sound is auditory. If you are asking about "
                   "synesthesia (where people experience sounds as colours), that is "
                   "a different question with individually varying answers."),
    },
    {
        "type": "impossible",
        "q": "How many sides does a circle have?",
        "flaw": ("A circle is a continuous curve with no sides or corners in the "
                 "polygon sense."),
        "think": ("'Sides' in the polygon sense means straight line segments between "
                  "vertices. A circle has no straight segments and no vertices. "
                  "The question applies a polygon concept to a non-polygon shape."),
        "answer": ("A circle has no sides in the geometric sense — it is a continuous "
                   "curve, not a polygon. If 'sides' means something else (e.g. 'an "
                   "infinite number of infinitesimal sides'), that is a different and "
                   "non-standard framing that should be stated explicitly."),
    },
    {
        "type": "impossible",
        "q": "What happens when an unstoppable force meets an immovable object?",
        "flaw": ("The premise is self-contradictory: an unstoppable force and an "
                 "immovable object cannot coexist in the same universe by definition."),
        "think": ("If an unstoppable force exists, no object can be immovable "
                  "(since the force would move it). If an immovable object exists, "
                  "no force can be unstoppable. The premise requires both simultaneously, "
                  "which is a logical contradiction. The question has no physical answer."),
        "answer": ("The scenario is a logical paradox — the two objects are mutually "
                   "exclusive by definition. A universe with an unstoppable force "
                   "cannot also contain an immovable object. The question presupposes "
                   "a contradiction and has no physical answer."),
    },
    {
        "type": "impossible",
        "q": "Can you tell me the last digit of pi?",
        "flaw": "Pi is irrational and has no last digit — its decimal expansion is infinite.",
        "think": ("Pi (π ≈ 3.14159…) is an irrational number with a non-terminating, "
                  "non-repeating decimal expansion. It has no last digit. Asking for "
                  "a last digit presupposes the expansion terminates, which it does not."),
        "answer": ("Pi has no last digit — it is an irrational number whose decimal "
                   "expansion goes on forever without repeating. The premise that "
                   "there is a last digit is false."),
    },
    # --- categorical ---
    {
        "type": "categorical",
        "q": "How much does the number 7 weigh?",
        "flaw": "Abstract numbers have no physical mass.",
        "think": ("Weight is a physical property requiring mass in a gravitational "
                  "field. The number 7 is an abstract mathematical object with no "
                  "physical instantiation. Asking its weight applies a physical "
                  "attribute to an abstract entity."),
        "answer": ("The number 7 has no weight — it is an abstract mathematical "
                   "concept, not a physical object. Physical quantities like weight "
                   "do not apply to numbers."),
    },
    {
        "type": "categorical",
        "q": "What is the square root of red?",
        "flaw": "Colours are not numbers; square roots require numerical input.",
        "think": ("Square root is a mathematical operation defined for numbers. "
                  "'Red' is a colour (a perceptual/wavelength phenomenon) and is "
                  "not a number unless a specific numerical encoding (e.g. hex RGB) "
                  "is specified. The question applies a numerical operation to a "
                  "non-numerical entity."),
        "answer": ("'Red' is not a number, so it has no square root in the standard "
                   "sense. If you mean the square root of the RGB hex code for red "
                   "(e.g. #FF0000 = 16711680 decimal), that is a different and "
                   "specific question."),
    },
    {
        "type": "categorical",
        "q": "What does justice smell like?",
        "flaw": "Abstract concepts like justice have no olfactory properties.",
        "think": ("Justice is an abstract normative concept with no physical form "
                  "or sensory properties. Asking what it smells like applies a "
                  "sensory attribute (smell, an olfactory phenomenon) to an abstract "
                  "entity. Unless metaphor is the explicit intent, the question "
                  "contains a category error."),
        "answer": ("Justice is an abstract concept and has no smell — olfactory "
                   "properties require a physical substance. If you are asking "
                   "metaphorically (e.g. for poetic or rhetorical purposes), "
                   "that is a different and valid question, but the literal "
                   "answer is: it has none."),
    },
    {
        "type": "categorical",
        "q": "Is the number 3 taller than the number 5?",
        "flaw": "Numbers have no height; height is a spatial property of physical objects.",
        "think": ("Height is a physical measurement requiring a spatial extent. "
                  "Numbers are abstract and have no height. The comparison 'taller' "
                  "applies a physical attribute to mathematical abstractions."),
        "answer": ("Neither 3 nor 5 has a height — numbers are abstract objects "
                   "without physical dimensions. The comparison 'taller' does not "
                   "apply to numbers. (If you mean the printed digits, that depends "
                   "on the font.)"),
    },
    # --- selfcontradiction ---
    {
        "type": "selfcontradiction",
        "q": "Is the statement 'this statement is false' true or false?",
        "flaw": "This is the Liar paradox — a self-referential contradiction with no truth value.",
        "think": ("If the statement is true, then what it says ('this statement is "
                  "false') is true, making it false — contradiction. If it is false, "
                  "then 'this statement is false' is false, making it true — "
                  "contradiction. The statement is a paradox with no consistent "
                  "truth value in classical logic."),
        "answer": ("This is the Liar paradox. In classical two-valued logic it has "
                   "no consistent truth value — calling it true leads to false, "
                   "calling it false leads to true. The sentence is self-referentially "
                   "contradictory, not a well-formed truth-apt statement."),
    },
    {
        "type": "selfcontradiction",
        "q": "What is a word that means the same as 'unique' but is more unique?",
        "flaw": "'Unique' means one of a kind and admits no degrees of comparison.",
        "think": ("'Unique' means 'the only one of its kind'. As an absolute property, "
                  "it does not admit degrees — something is either unique or not. "
                  "'More unique' is a contradiction: if something is unique, nothing "
                  "can be more so. The premise assumes uniqueness is gradable, which "
                  "it is not in standard usage."),
        "answer": ("'Unique' is an absolute adjective: it means 'the only one of its "
                   "kind' and has no degrees. 'More unique' is self-contradictory — "
                   "if something is unique, nothing can be more so. A synonym that "
                   "is 'more unique' is not a coherent request."),
    },
    {
        "type": "selfcontradiction",
        "q": "Can you give me a rule that has no exceptions, including no exception to itself?",
        "flaw": ("'No exceptions' applied to itself creates a paradox if the rule has "
                 "any exception at all."),
        "think": ("A rule with 'no exceptions' that is itself exception-free would "
                  "need to cover all possible cases perfectly. But asking for a rule "
                  "with no exception to itself is either trivially satisfied (every "
                  "rule either has exceptions or does not) or leads to a self-reference "
                  "issue similar to the Liar paradox."),
        "answer": ("The request is self-referentially problematic. A rule can state "
                   "'this rule has no exceptions', which is self-consistent only if "
                   "it is actually universally true. If even one exception exists, "
                   "including an exception to the 'no exceptions' claim, the rule "
                   "is false. Most such rules either are trivially true in limited "
                   "domains or fail when examined broadly."),
    },
    # --- counterfactual ---
    {
        "type": "counterfactual",
        "q": "Given that Napoleon won the Battle of Waterloo, what did he do next?",
        "flaw": "Napoleon lost the Battle of Waterloo in 1815.",
        "think": ("The premise 'Napoleon won the Battle of Waterloo' is historically "
                  "false. Napoleon was defeated at Waterloo on 18 June 1815, leading "
                  "to his final abdication and exile to Saint Helena. I should "
                  "correct this before answering."),
        "answer": ("Napoleon did not win the Battle of Waterloo — he was decisively "
                   "defeated there on 18 June 1815 by the Duke of Wellington and "
                   "Prussian forces. This led to his second abdication and exile "
                   "to Saint Helena, where he died in 1821. If you are asking about "
                   "a counterfactual history, please frame it as such."),
    },
    {
        "type": "counterfactual",
        "q": "Since the Earth is flat, why do ships disappear over the horizon?",
        "flaw": "The Earth is not flat; it is an oblate spheroid.",
        "think": ("The premise 'the Earth is flat' is false. The Earth is an oblate "
                  "spheroid, confirmed by centuries of observation, GPS, satellite "
                  "imagery, and physics. The 'disappear over the horizon' phenomenon "
                  "is actually evidence for Earth's curvature, not the flat-Earth "
                  "premise. I should correct the premise."),
        "answer": ("The premise is false: the Earth is not flat. It is an oblate "
                   "spheroid. Ships disappear hull-first over the horizon because "
                   "Earth's curved surface drops away — this is direct evidence "
                   "for a spherical Earth, not consistent with a flat one."),
    },
    {
        "type": "counterfactual",
        "q": "Why did Einstein fail his mathematics exams in school?",
        "flaw": "Einstein did not fail mathematics; he excelled at it.",
        "think": ("This is a well-known myth. Einstein was exceptionally strong in "
                  "mathematics from an early age and had studied calculus by 15. "
                  "The myth originates from a misreading of Swiss grading scales. "
                  "He did fail one entrance exam at 15 (in non-mathematical subjects), "
                  "but not mathematics. I should correct the false premise."),
        "answer": ("Einstein did not fail mathematics exams — the premise is false. "
                   "He excelled at mathematics from childhood. He did fail an "
                   "entrance exam at 15, but in French and botany, not mathematics. "
                   "The myth comes from misreading Swiss grading where 6 is the "
                   "top mark, not 1."),
    },
    {
        "type": "counterfactual",
        "q": "Since humans only use 10% of their brain, what does the other 90% do?",
        "flaw": "The '10% of the brain' claim is a myth; humans use virtually all of their brain.",
        "think": ("The premise that humans use only 10% of their brain is a widely "
                  "repeated myth with no scientific basis. Neuroimaging shows activity "
                  "throughout the brain, different regions handle different functions, "
                  "and brain damage anywhere causes functional deficits — not consistent "
                  "with 90% being unused. I should correct this before engaging."),
        "answer": ("The '10% of the brain' claim is a myth. Humans use virtually "
                   "all of their brain — different regions are active at different "
                   "times depending on the task, and neuroimaging shows no large "
                   "unused regions. Brain damage anywhere causes measurable deficits. "
                   "There is no unused 90%."),
    },
    # --- arithmetic ---
    {
        "type": "arithmetic",
        "q": "If a product costs £10 and is discounted by 120%, what is the final price?",
        "flaw": "A discount of 120% would make the price negative, which is not physically meaningful for a standard sale.",
        "think": ("A 120% discount on a £10 item would be £12 off, giving −£2. "
                  "A negative price means the seller would pay the buyer £2. "
                  "While mathematically computable, a discount exceeding 100% is "
                  "not a standard commercial scenario and the premise likely contains "
                  "an error (perhaps 20% was intended)."),
        "answer": ("A 120% discount on £10 would mathematically yield −£2 (the seller "
                   "pays the buyer £2). This is not a meaningful commercial scenario "
                   "— discounts above 100% cannot occur in a standard sale. The "
                   "premise likely contains an error; if the discount is 20%, the "
                   "final price is £8."),
    },
    {
        "type": "arithmetic",
        "q": "A train travels at 200 km/h for 30 minutes. It covers 200 km. Is this correct?",
        "flaw": "200 km/h for 30 minutes (0.5 hours) covers 100 km, not 200 km.",
        "think": ("Distance = speed × time. 200 km/h × 0.5 h = 100 km. The stated "
                  "distance of 200 km is incorrect by a factor of 2. The premise "
                  "contains an arithmetic error."),
        "answer": ("No, the calculation is incorrect. At 200 km/h for 30 minutes "
                   "(0.5 hours), the train covers 200 × 0.5 = 100 km, not 200 km. "
                   "200 km would require 1 full hour at that speed."),
    },
    {
        "type": "arithmetic",
        "q": "There are 7 days in a week and 4 weeks in a month. So a year has exactly 336 days — agree?",
        "flaw": "7×4×12 = 336, but months have more than 4 weeks; a year has 365/366 days.",
        "think": ("The calculation 7×4×12 = 336 follows from the stated premises, "
                  "but the premise '4 weeks in a month' is wrong. Months have "
                  "28–31 days (4 to 4.43 weeks). A year has 365 or 366 days. "
                  "The approximate '4 weeks per month' loses 29–30 days annually."),
        "answer": ("The arithmetic 7×4×12 = 336 is correct, but the premise is "
                   "wrong: months do not have exactly 4 weeks. They have 28–31 days. "
                   "A year has 365 or 366 days, not 336. The '4 weeks per month' "
                   "approximation loses about 29 days per year."),
    },
]


def _think_text(item: dict) -> str:
    return (
        f"Let me check the premise before reasoning.\n\n"
        f"Premise check: {item['flaw']}\n\n"
        f"{item['think']}\n\n"
        f"I should name the flaw in my answer rather than reason from the "
        f"false premise."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    items = list(ITEMS)
    if args.n > len(items):
        while len(items) < args.n:
            items.extend(ITEMS)
    random.shuffle(items)
    items = items[:args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for i, item in enumerate(items):
            rec = {
                "id": f"premise_{item['type']}_{i:03d}",
                "system": SYSTEM,
                "user": item["q"],
                "think": _think_text(item),
                "answer": item["answer"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written {len(items)} premise-refusal items → {out}")


if __name__ == "__main__":
    main()
