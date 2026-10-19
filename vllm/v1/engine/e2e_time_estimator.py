# SPDX-License-Identifier: Apache-2.0


class E2ETimeEstimator:

    def estimate(self, prefill_time: float, output_length: int) -> float:
        # TODO: Implement custom logic
        return prefill_time + output_length * 0.01
