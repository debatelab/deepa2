# Entailment Bank to DeepA2

## Original Dataset

Q/A dataset with proofs for a `hypothesis`, which explains the correct `answer` to a `question`. We're using subsets `task_1` (no distractors) and `task_2` (up to two dozens of distracting statements).

### Features (selection)

```yaml
question: 'Many stars can be seen in the sky at night. 
   Which statement best explains why the Sun appears 
   brighter than the stars seen in the night sky?'
answer: 'The Sun is closer to Earth than the stars 
   seen in the night sky.'
hypothesis: 'the sun is brighter than other stars 
   because the sun is closer to earth than other stars'
core_concepts:
-  'as a source of light becomes closer , the light 
   will appear brighter'
step_proof: 'sent10 & sent20 -> int1: stars are a source 
   of light; int1 & sent7 -> int2: as the stars becomes
   closer, the light of the stars will appear brighter;
   sent15 & sent5 -> int3: the sun is closer to earth than
   other stars to earth; int2 & int3 -> hypothesis;'
distractors: 
-  'sent4'
-  'sent6'
-  ...
triples:
-  sent1: 'being in the sun is synonymous with being 
      in the sunlight'
-  sent2: 'amount is a property of something and includes
      ordered values of none / least / little / some / half 
      / much / many / most / all'
-  sent3: 'distant means great in distance'
-  sent4: ...
intermediate_conclusions: 
-  int1: 'stars are a source of light'
-  int2: 'as the stars becomes closer, the light of the 
      stars will appear brighter'
-  int3: ...  
```


### Source

* [entailment bank (Allen AI)](https://allenai.org/data/entailmentbank)
* [entailment bank (github)](https://github.com/allenai/entailment_bank/)

### License

CC BY 4.0

### Citation

```bibtex
@article{Dalvi2021ExplainingAW,
  title={Explaining Answers with Entailment Trees},
  author={Bhavana Dalvi and Peter Alexander Jansen and Oyvind Tafjord and Zhengnan Xie and Hannah Smith and Leighanna Pipatanangkura and Peter Clark},
  journal={ArXiv},
  year={2021},
  volume={abs/2104.08661}
}
```

## Preprocessing Entailment Bank for DeepA2

Preprocessing expands meta-data and drops all columns not specified above. 

## DeepA2-ENBANK

### Construction

From each preprocessed Entailment Bank example, we build a single DeepA2 item.

First, we render the proof chain as an argdown argument.

Second, we construct an argument source text of the form 

```
{{ question_text }} {{ answer_text }}.
that is because {{ statements }}
```

while shuffling sentences (explicit premises and distractors). All intermediary conclusions are implicit. 

Thirdly, we construct lists of reasons and conjectures, and a clear paraphrase of the source text (which drops all distractors). We use `hypothesis` as the argument's `gist`, a `core_concepts` as the its `title`, and `question` as its `context`.


### Features

- [x] `source_text`
- [x] `title`
- [x] `gist`
- [x] `source_paraphrase`
- [x] `context`

<!-- -->

- [x] `reasons`
- [x] `conjectures`

<!-- -->

- [x] `argdown_reconstruction`
- [ ] `erroneous_argdown`
- [x] `premises`
- [ ] `intermediary_conclusion`
- [x] `conclusion`

<!-- -->

- [ ] `premises_formalized`
- [ ] `intermediary_conclusion_formalized`
- [ ] `conclusion_formalized`
- [ ] `predicate_placeholders`
- [ ] `entity_placeholders`
- [ ] `misc_placeholders`
- [ ] `plchd_substitutions`




