# SPDX-License-Identifier: Apache-2.0

import dataclasses


@dataclasses.dataclass
class UserMetadata:

    # User identifier. Used to group requests per users in multi-user scenarios
    uid: str

    # Latency SLO in seconds
    slo: float

    # Attacker
    is_attacker: bool = False
