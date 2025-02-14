# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark2.benchmark_method import BenchmarkMethod
from ax.exceptions.core import UserInputError
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.scheduler import SchedulerOptions
from ax.utils.common.testutils import TestCase


class TestBenchmarkMethod(TestCase):
    def test_benchmark_method(self):
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=10,
                )
            ],
            name="SOBOL",
        )
        options = SchedulerOptions(total_trials=10)
        method = BenchmarkMethod(
            name="Sobol10", generation_strategy=gs, scheduler_options=options
        )

        self.assertEqual(method.generation_strategy, gs)
        self.assertEqual(method.scheduler_options, options)

    def test_total_trials_none(self):
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=10,
                )
            ],
            name="SOBOL",
        )
        options = SchedulerOptions()

        with self.assertRaisesRegex(
            UserInputError, "SchedulerOptions.total_trials may not be None"
        ):
            BenchmarkMethod(
                name="Sobol10", generation_strategy=gs, scheduler_options=options
            )
