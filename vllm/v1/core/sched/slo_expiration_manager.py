# SPDX-License-Identifier: Apache-2.0

import time

from vllm.v1.request import Request


class SLOExpirationManager:

    _URGENT = -10000000000
    _NO_PRIORITY = 10000000000

    def update_priorities(self, requests: dict[str,
                                               Request]) -> dict[str, Request]:
        now = time.time()
        for request in requests.values():
            if request.request_md and request.user_md and \
                request.request_md.estimated_time > 0 and \
                request.user_md.slo > 0 and \
                request.request_md.estimated_time > request.user_md.slo:
                request.priority = self._URGENT
                continue

            if request.user_md and request.user_md.slo > 0:
                time_passed = now - request.arrival_time
                deadline_in = request.user_md.slo - time_passed
                request.priority = deadline_in * 1000000
            else:
                request.priority = self._NO_PRIORITY

        return requests
