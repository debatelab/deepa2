# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Basic metrics for assessing generated argument
reconstructions in the DA2 framework.
"""

import datasets

from deepa2.metrics import (
    DA2PredictionEvaluator,
)


_CITATION = """\
@article{Betz2021DeepA2AM,
	author = {Gregor Betz and Kyle Richardson},
	date-added = {2022-04-13 14:52:32 +0200},
	date-modified = {2022-04-13 14:52:32 +0200},
	journal = {ArXiv},
	title = {DeepA2: A Modular Framework for Deep Argument Analysis
    with Pretrained Neural Text2Text Language Models},
	volume = {abs/2110.01509},
	year = {2021}
}
"""

_DESCRIPTION = """\
DeepA2 metrics for assessing the basic quality of argument reconstructions.
"""


_KWARGS_DESCRIPTION = """
Calculates how good are argument reconstructions, using basic argument quality metrics.
Args:
    predictions: list of predictions to score.
    references: list of reference for each prediction.
Returns:
    valid_argdown: is the argument set in valid argdown format?,
    pc_structure: does the argument have a valid premise-conclusion structure?,
    consistent_usage: is every premise used in an inference?,
    no_petitio: the argument is not a petitio,
    no_redundancy: no statement occurs twice in the argument,
    inferential_similarity: extent to which the argument is similar to the reference,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> my_new_metric = datasets.load_metric("da2_metric")
    >>> ref = "(1) P ---- (2) C"
    >>> prd = "(1) P ---- (2) C"
    >>> results = my_new_metric.compute(references=[ref], predictions=[prd])
    >>> print(results)
"""



@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class DA2Metric(datasets.Metric):
    """
    Basic metrics for assessing generated argument
    reconstructions in the DA2 framework.
    """

    def _info(self):
        # Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="https://github.com/debatelab/deepa2",
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/debatelab/deepa2"],
            reference_urls=["https://arxiv.org/abs/2110.01509"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        # Download external resources if needed
        # bad_words_path = dl_manager.download_and_extract(BAD_WORDS_URL)
        # self.bad_words = {w.strip() for w in open(bad_words_path, encoding="utf-8")}
        self.scorer = DA2PredictionEvaluator()  # pylint: disable=attribute-defined-outside-init

    def _compute(self, predictions, references):  # pylint: disable=arguments-differ
        """Returns the scores"""

        metrics = self.scorer.compute_metrics(predictions, references)

        return metrics
