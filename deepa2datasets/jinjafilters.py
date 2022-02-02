"""Jinja2 filters used for constructing DeepA2 items from eSNLI data."""

import random

templates_conditional = [
    "if {antecent} then {consequent}.",
    "presuming that {antecent}, {consequent}.",
    "{consequent} if {antecent}.",
    "{consequent} provided that {antecent}.",
    "{antecent} only if {consequent}.",
]

templates_negation = [
    "it is not the case that {sentence}.",
    "it is wrong that {sentence}.",
    "it is false that {sentence}.",
]


# strip and lower case
def lowerall(p):
    p = p.lower()
    return p

def sal(p):
    p = p.strip(" .")
    p = lowerall(p)
    return p

#negate
def negation(p):
    t = random.choice(templates_negation)
    p = t.format(sentence=sal(p))
    return p

# create conditional
def conditional(antecedent,consequent):
    t = random.choice(templates_conditional)
    p = t.format(antecent=sal(antecedent),consequent=sal(consequent))
    return p