# IBM-ArgQ to DeepA2

## Original Dataset

30,497 `arguments` for 71 `topics` labeled for quality and `stance`.

### Features (selection)

```yaml
topic: 'we should ban cosmetic surgery for minors'
argument: 'a birth defect or a disfiguring injury can highly impact someone's
self esteem, with parental permission minors should have the opportunity to
correct this'
stance: -1
```


### Source

* [ArgQ (IBM)](https://research.ibm.com/haifa/dept/vst/debating_data.shtml)


### License

The datasets are released under the following licensing and copyright terms:

* (c) Copyright Wikipedia (https://en.wikipedia.org/wiki/Wikipedia:Copyrights#Reusers.27_rights_and_obligations)
* (c) Copyright IBM 2014. Released under CC-BY-SA (http://creativecommons.org/licenses/by-sa/3.0/)


### Citation

```bibtex
@article{GretzArgQ2020,
  title={A Large-scale Dataset for Argument Quality Ranking: Construction and Analysis},
  author={Shai Gretz, Roni Friedman, Edo Cohen-Karlik, Assaf Toledo, Dan Lahav, Ranit Aharonov and Noam Slonim},
  journal={AAAI},
  year={2020}
}
```

## Preprocessing ArgQ for DeepA2

Preprocessing adds a relevant `opposing argument` (same topic, opposite stance) and drops all columns not specified above. 

## DeepA2-ARGQ

### Construction

From each preprocessed ArgQ example, we build a single DeepA2 item, including the original `argument` and the `opposing argument` added as a distractor. For example:

```yaml
source_text: "we should ban cosmetic surgery for minors? I disagree! cosmetic
surgery is dangerous for all ages but especially so for minors therefore it
should be banned for their age. a birth defect or a disfiguring injury can
highly impact someone&#39;s self esteem, with parental permission minors should
have the opportunity to correct this." 
reasons:
   - text: "a birth defect or a disfiguring injury can highly impact someone's
   self esteem, with parental permission minors should have the opportunity to
   correct this"
conjectures:
   - text: "we should ban cosmetic surgery for minors? I disagree!"
```


### Features

- [x] `source_text`
- [ ] `title`
- [ ] `gist`
- [ ] `source_paraphrase`
- [ ] `context`

<!-- -->

- [x] `reasons`
- [x] `conjectures`

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




