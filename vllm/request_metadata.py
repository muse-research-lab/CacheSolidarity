# SPDX-License-Identifier: Apache-2.0

import dataclasses


@dataclasses.dataclass
class RequestMetadata:

    # Estimated end-to-end latency in seconds
    estimated_time: float
