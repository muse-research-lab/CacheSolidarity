# SPDX-License-Identifier: Apache-2.0

from vllm.v1.engine import EngineCoreRequest


class PrefillTimeEstimator:

    def estimate(self, request: EngineCoreRequest) -> float:
        # TODO: Implement custom logic
        num_prompt_tokens = len(request.prompt_token_ids)
        return num_prompt_tokens * 0.005
