# IBM-KPA to DeepA2 (`argkp`)

## Original Dataset

~24K argument/key-point pairs, for 28 controversial topics. Each of the pairs is labeled as matching/non-matching, as well as assigned a stance towards the topic. 

### Features (selection)

```yaml
topic: 'we should end affirmative action'
argument: 'affirmative action forces employers to pick people based on features not merits'
stance: 1
key_point: 'affirmative action reduces quality'
```


### Source

* [KPA (IBM)](https://github.com/IBM/KPA_2021_shared_task)


### License

The datasets are released under the following licensing and copyright terms:

* Apache-2.0 license


### Citation

```bibtex
@misc{BarHaim2020,
  doi = {10.48550/ARXIV.2010.05369},
  url = {https://arxiv.org/abs/2010.05369},
  author = {Bar-Haim, Roy and Kantor, Yoav and Eden, Lilach and Friedman, Roni and Lahav, Dan and Slonim, Noam},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Quantitative Argument Summarization and Beyond: Cross-Domain Key Point Analysis},
  publisher = {arXiv},
  year = {2020},
}
```

## Preprocessing KPA for DeepA2

Merges data that is originally provided in three files into a single dataset. 

## DeepA2-ARGKP

### Construction

From each preprocessed KPA example, we build a single DeepA2 item. The `topic` and `argument` serve as building blocks for the source_text, the `key_point` is the main premise in the reconstructed argument, from which the conclusion (derived from `topic` and `stance`) is inferred with modus ponens. For example:

```yaml
source_text: "we should end affirmative action? affirmative action forces employers to pick people based on features not merits." 
argdown_reconstruction: "(1) affirmative action reduces quality. (2) if affirmative action reduces quality then we should end affirmative action. -- with modus ponens from (1) (2) -- (3) we should end affirmative action."
premises:
   -  text: "affirmative action reduces quality"
      ref_reco: 1
   -  text: "if affirmative action reduces quality then we should end affirmative action"
      ref_reco: 2
conclusion:
   -  text: "we should end affirmative action"
      ref_reco: 3
context: "we should end affirmative action? I agree!"
gist: "affirmative action reduces quality."
premises_formalized:
   -  text: "p"
      ref_reco: 1
   -  text: "p -> q"
      ref_reco: 2
conclusion_formalized:
   -  text: "q"
      ref_reco: 3
plchd_substitutions: 
   -  -  "p"
      -  "affirmative action reduces quality"
   -  -  "q"
      -  "we should end affirmative action"
```


### Features

- [x] `source_text`
- [ ] `title`
- [x] `gist`
- [ ] `source_paraphrase`
- [x] `context`

<!-- -->

- [ ] `reasons`
- [ ] `conjectures`

<!-- -->

- [x] `argdown_reconstruction`
- [ ] `erroneous_argdown`
- [x] `premises`
- [ ] `intermediary_conclusion`
- [x] `conclusion`

<!-- -->

- [x] `premises_formalized`
- [ ] `intermediary_conclusion_formalized`
- [x] `conclusion_formalized`
- [ ] `predicate_placeholders`
- [ ] `entity_placeholders`
- [x] `misc_placeholders`
- [x] `plchd_substitutions`




