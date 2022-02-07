# AIFdb to DeepA2

## Original Dataset

AIFdb is an online databank for argument analyses run by the [Centre for Argument Technology](https://arg-tech.org/) at the University of Dundee. It is organized in corpora which comprise individual argument maps. Maps can be browsed online via the [OVA editor](https://arg-tech.org/index.php/ova/).

### Features

Each argument map in a AIFdb corpus comprises 

* a `source text` and 
* a `nodeset` with argumentative annotatations of the `source text`.

The `nodeset` identifies individual claims in the `source text` and describes their argumentative relations, essentially **argumentative support** and **conflict**. These annotations are stored as intertextual correspondences [[Visser et al. 2018]](http://aclanthology.lst.uni-saarland.de/L18-1554.pdf).

### Source

* [AIFdb](https://www.aifdb.org/)

### License

Free for academic use ([more info](https://arg-tech.org/index.php/research/argument-corpora/)).

### Citation

```bibtex
@inproceedings{lawrence2014aifdb,
  title={AIFdb Corpora.},
  author={Lawrence, John and Reed, Chris},
  booktitle={COMMA},
  pages={465--466},
  year={2014}
}
```

## Preprocessing AIFdb for DeepA2

Each argument in a AIFdb argument map gives rise to a separate preprocessed example. More precisely, we extract all default inferential relations in a nodeset (default inference and default conflict); for each of these inferences, we identify premises, conclusion and the corresponding annotated sequences in the `source text`. In addition, a single preprocessed example from AIFdb comprises information about the type of inferential relation and the original corpus, e.g.:

```yaml
text: "MB: Melanie Phillips? MP: You say in this case there 
    are specific victims. Isn't the problem though that there 
    are not very specific perpetrators? Why should the present 
    government of Britain be held responsible for 
    responsibility for the these atrocities of its predecessor 
    administration many decades ago? ES: I disagree that 
    there were not specific perpetrators. Obviously we cannot 
    for legal reasons go into sort of identifying individuals, 
    why the current government of today should be held liable 
    is because there is an outstanding crime that has been 
    committed, these were not only, you know, crimes of torture, 
    there were war crimes committed, there were crimes against 
    humanity. My understanding of international law is that 
    there is no statute of limitations, I think it would be 
    totally different if the survivors had all died. But 
    actually, you have living victims."
corpus: "Moral Maze British Empire"
type: "Default Inference"
premises:
-  "there were war crimes committed,"
-  "these were not only crimes of torture"
conclusion:
-  "the current government of today should be held liable"
reasons:
-  "ES : there were war crimes committed,"
-  "ES : these were not only, you know, crimes of torture"
conjectures:
-  "ES : why the current government of today should be held liable"
```

(In some cases, no text is provided by AIFdb. If so, we concatenate all annotations provided by the `nodeset`.)

## DeepA2-AIFdb

### Construction

From each preprocessed AIFdb example, we build a single DeepA2 item. It's straightforward to cast source text, reasons and conjectures to AIFdb. Moreover, we use premises and conclusion to compile a clear and succinct paraphrase of the reconstructed argument (in function of the type of inferential relation between premises and conclusion).


### Features

- [x] `argument_source`
- [ ] `title`
- [ ] `gist`
- [x] `source_paraphrase`
- [ ] `context`

<!-- -->

- [x] `reasons`
- [x] `conjectures`
- [ ] `distractors`

<!-- -->

- [ ] `argdown_reconstruction`
- [ ] `erroneous_argdown`
- [ ] `premises`
- [ ] `intermediary_conclusion`
- [ ] `conclusion`

<!-- -->

- [ ] `premises_formalized`
- [ ] `intermediary_conclusion_formalized`
- [ ] `conclusion_formalized`
- [ ] `predicate_placeholders`
- [ ] `entity_placeholders`
- [ ] `misc_placeholders`
- [ ] `plchd_substitutions`




