# e-SNLI to DeepA2

## Original Dataset

NLI dataset with explanations for why the inferential relations indicated by the `label` holds between `premise` and `hypothesis`.

### Features

```json
{
    "premise":"This church choir sings to the masses as 
        they sing joyous songs from the book at a church.",
    "hypothesis":"The church has cracks in the ceiling.",
    "label":"neutral",
    "explanation_1":"Not all churches have cracks in the 
        ceiling",
    "explanation_2":"There is no indication that there are
        cracks in the ceiling of the church.",
    "explanation_3":"Not all churches have cracks in the 
        ceiling."
}
```



### Source

* [esnli at github](https://github.com/OanaMariaCamburu/e-SNLI)
* [esnli at hugging face](https://huggingface.co/datasets/esnli)

### Licence

MIT License

### Citation

```bibtex
@incollection{NIPS2018_8163,
  title = {e-SNLI: Natural Language Inference with Natural Language Explanations},
  author = {Camburu, Oana-Maria and Rockt\"{a}schel, Tim and Lukasiewicz, Thomas and Blunsom, Phil},
  booktitle = {Advances in Neural Information Processing Systems 31},
  editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
  pages = {9539--9549},
  year = {2018},
  publisher = {Curran Associates, Inc.},
  url = {http://papers.nips.cc/paper/8163-e-snli-natural-language-inference-with-natural-language-explanations.pdf}
}
```

## Preprocessing e-SNLI

e-SNLI examples are grouped by `premise` and splitted into chunks of three items with different labels (but identical premise). These three items are then merged into a single preprocessed example, e.g.:

```json
{
    "premise":"This church choir sings to the masses as 
        they sing joyous songs from the book at a church.",
    "hyp_neutral":"The church has cracks in the ceiling.",
    "hyp_entailm":"The church is filled with song.",
    "hyp_contrad":"A choir singing at a baseball game.",
    "expl_neutral":[
        "Not all churches have cracks in the ceiling",
        "There is no indication that there are cracks 
        in the ceiling of the church.",
        "Not all churches have cracks in the ceiling."
    ],
    "expl_entailm":[
        "'Filled with song' is a rephrasing of the 'choir 
        sings to the masses.'",
        "hearing song brings joyous in the church.",
        "If the church choir sings then the church is 
        filled with song."
    ],
    "expl_contrad":[
        "A choir sing some other songs other than book 
        at church during the base play. they cannot see 
        book and play base ball same time.",
        "The choir is at a chruch not a baseball game.",
        "A baseball game isn’t played at a church."
    ]
}
```

## DeepA2-ESNLI

### Construction

From each preprocessed eSNLI example, we build multiple DeepA2 items.

First, we can build simple propositional arguments (including formalizations) of the form

```
(1) premise
(2) if premise, then hyp_entailm
-- with modus ponens from 1,2 --
(3) hyp_entailment
```

or, respectively:

```
(1) premise
(2) if hyp_contrad, then not premise
-- with modus tollens from 1,2 --
(3) not hyp_contrad
```

Second, we construct argument source texts by shuffling premises, hypotheses and explanations. In doing so, some of the sentences serve as distractors, others may be left out to the effect that a premise and or the conclusion are implicit. 

Thirdly, we construct lists of reasons and conjectures, which identify explicitly stated parts of the argument and map them to their counterparts in the argdown snippet.


### Features

- [x] argument_source
- [ ] title
- [x] gist
- [x] source_paraphrase
- [ ] context

<!-- -->

- [x] reason_statements
- [x] conclusion_statements
- [x] distractors

<!-- -->

- [x] argdown_reconstruction
- [x] erroneous_argdown
- [x] premises
- [ ] intermediary_conclusion
- [x] conclusion

<!-- -->

- [x] premises_formalized
- [ ] intermediary_conclusion_formalized
- [x] conclusion_formalized
- [ ] predicate_placeholders
- [ ] entity_placeholders
- [x] misc_placeholders




