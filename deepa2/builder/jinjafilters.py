"""Jinja2 filters used for constructing DeepA2 items from eSNLI data."""

import random

templates_conditional = [
    "if {antecent} then {consequent}.",
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


def lowerall(sentence: str):
    """to lower case"""
    sentence = sentence.lower()
    return sentence


def sal(sentence: str):
    """strip"""
    sentence = sentence.strip(" .")
    sentence = lowerall(sentence)
    return sentence


def negation(sentence: str):
    """negate"""
    template = random.choice(templates_negation)
    sentence = template.format(sentence=sal(sentence))
    return sentence


def conditional(antecedent: str, consequent: str):
    """create conditional"""
    template = random.choice(templates_conditional)
    sentence = template.format(antecent=sal(antecedent), consequent=sal(consequent))
    return sentence
