# Copyright 2026 Srikumar Krishnamoorthy
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

"""hugiml.benchmarks — reproducible HUG-IML benchmark comparison suite.

Provides a cross-validated benchmark runner that compares HUGIMLClassifierNative
against EBM, XGBoost, Random Forest, Logistic Regression, RuleFit, and GAM
on standard tabular datasets.

Console entry point (installed as ``hugiml-bench``):

    hugiml-bench --datasets breast_cancer adult --n-splits 5 --output results/

Or run as a Python module:

    python -m hugiml.benchmarks.runner --datasets breast_cancer adult
"""

from hugiml.benchmarks.runner import main

__all__ = ["main"]
