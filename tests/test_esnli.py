from __future__ import annotations

import pytest
import datasets

from deepa2datasets.nli_builder import eSNLIBuilder, RawESNLIExample, PreprocessedESNLIExample

RAW_EXAMPLES_1 = [
    RawESNLIExample(
        premise="premise",
        hypothesis="he",
        label=0,
        explanation_1="ee1",
        explanation_2="ee2",
        explanation_3="ee3"
    ),
    RawESNLIExample(
        premise="premise",
        hypothesis="hn",
        label=1,
        explanation_1="en1",
        explanation_2="en2",
        explanation_3="en3"
    ),
    RawESNLIExample(
        premise="premise",
        hypothesis="hc",
        label=2,
        explanation_1="ec1",
        explanation_2="ec2",
        explanation_3="ec3"
    )
]

PREPROCESSED_EXAMPLES_1 = [
    PreprocessedESNLIExample(
        premise="premise",
        hypothesis_ent="he",
        hypothesis_neu="hn",
        hypothesis_con="hc",
        explanation_ent=["ee1","ee2","ee3"],
        explanation_neu=["en1","en2","en3"],
        explanation_con=["ec1","ec2","ec3"]
    )
]


def test_esnli_preprocessor():
    """tests esnli preprocessor"""
    raw_data = {}
    for key in RawESNLIExample.__annotations__.keys(): # pylint: disable=no-member}
        raw_data[key] = [example[key] for example in RAW_EXAMPLES_1]
    dataset = datasets.Dataset.from_dict(raw_data)
    dataset = eSNLIBuilder.preprocess(dataset)
    processed_data = dataset.to_dict(batch_size=1,batched=True) # return iterator
    processed_examples = []
    for processed_example in processed_data:
        # unbatch
        processed_examples.append({k:v[0] for k,v in processed_example.items()})
    assert processed_examples == PREPROCESSED_EXAMPLES_1


@pytest.fixture
def processed_examples():
    """processes examples """
    da2items = []
    for preprocessed_example in PREPROCESSED_EXAMPLES_1:
        batched_input = {k:[v] for k,v in preprocessed_example.items()}
        builder = eSNLIBuilder()
        builder.input = batched_input
        builder.configure_product()
        builder.produce_da2item()
        builder.postprocess_da2item()
        builder.add_metadata_da2item()
        da2items.extend(builder.product) # product is a list of dicts
    return da2items


def test_esnli_conclusions_1(processed_examples):
    """contrary hypothesis is never conclusion"""
    conclusions = [da2item["conclusion"][0]["text"] for da2item in processed_examples]
    hyp_c = (PREPROCESSED_EXAMPLES_1[0].get("hypothesis_con"))
    print(conclusions)
    print(hyp_c)
    assert not hyp_c in conclusions

def test_esnli_conclusions_2(processed_examples):
    """neutral hypothesis is never conclusion"""
    conclusions = [da2item["conclusion"][0]["text"] for da2item in processed_examples]
    hyp_n = (PREPROCESSED_EXAMPLES_1[0].get("hypothesis_neu"))
    print(conclusions)
    print(hyp_n)
    assert not hyp_n in conclusions

def test_esnli_conclusions_3(processed_examples):
    """entailed hypothesis is sometimes conclusion"""
    conclusions = [da2item["conclusion"][0]["text"] for da2item in processed_examples]
    hyp_e = (PREPROCESSED_EXAMPLES_1[0].get("hypothesis_ent"))
    print(conclusions)
    print(hyp_e)
    assert (hyp_e in conclusions) and (len(set(conclusions))!=1)
