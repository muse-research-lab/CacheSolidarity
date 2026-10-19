# SPDX-License-Identifier: Apache-2.0

from collections import deque
from typing import Optional

from vllm.v1.request import Request


class QueueManager:

    def __init__(self):
        self.combined: list[Request] = []
        self.running_req_ids: set[str] = set()
        self.waiting_req_ids: set[str] = set()
        self.last_running_idx = None
        self.first_waiting_idx = None

    def get_last_running_idx(self) -> Optional[int]:
        return self.last_running_idx

    def num_running_requests(self) -> int:
        return len(self.running_req_ids)

    def reorder_running_queue(self, running: list[Request]) -> list[Request]:
        running = sorted(running, key=lambda req: req.priority or 0)
        return running

    def reorder_waiting_queue(self, waiting: deque[Request]) -> deque[Request]:
        waiting = deque(sorted(waiting, key=lambda req: req.priority or 0))
        return waiting

    def combine_queues(self, running: list[Request], waiting: deque[Request]):
        self.running_req_ids = set()
        self.waiting_req_ids = set()
        combined = []

        for req in running:
            combined.append(("running", req.priority, req))
            self.running_req_ids.add(req.request_id)

        for req in waiting:
            combined.append(("waiting", req.priority, req))
            self.waiting_req_ids.add(req.request_id)

        # Sort: by priority DESC, then by queue type ("running" < "waiting")
        combined.sort(key=lambda x: (x[1], x[0]))
        self.combined = [x[2] for x in combined]

        # Update last running idx
        self.last_running_idx = None
        for i in range(len(self.combined) - 1, -1, -1):
            if self.combined[i].request_id in self.running_req_ids:
                self.last_running_idx = i
                break

        # Update first waiting idx
        self.first_waiting_idx = None
        for i in range(len(self.combined)):
            if self.combined[i].request_id in self.waiting_req_ids:
                self.first_waiting_idx = i
                break

        return

    def update_queues(self) -> tuple[list[Request], deque[Request]]:
        if self.last_running_idx is not None and \
            self.first_waiting_idx is not None:
            assert self.last_running_idx == self.first_waiting_idx - 1
            return self.combined[:self.first_waiting_idx], deque(
                self.combined[self.first_waiting_idx:])

        elif self.last_running_idx is not None and \
            self.first_waiting_idx is None:
            return self.combined, deque([])

        elif self.last_running_idx is None and \
            self.first_waiting_idx is not None:
            return [], deque(self.combined)

        else:
            return [], deque([])

    def update_last_running_idx(self):
        if self.last_running_idx is None:
            return

        last_running_idx = self.last_running_idx

        # Update request id sets
        request_id = self.combined[last_running_idx].request_id
        self.running_req_ids.remove(request_id)
        self.waiting_req_ids.add(request_id)

        # Update last running idx
        self.last_running_idx = None
        for i in range(last_running_idx, -1, -1):
            if self.combined[i].request_id in self.running_req_ids:
                self.last_running_idx = i
                break

        # Update first waiting idx if necessary
        if self.last_running_idx is None and len(self.waiting_req_ids) > 0:
            self.first_waiting_idx = 0

        return

    def update_first_waiting_idx(self):
        if self.first_waiting_idx is None:
            return

        first_waiting_idx = self.first_waiting_idx

        # Update request id sets
        request_id = self.combined[first_waiting_idx].request_id
        self.waiting_req_ids.remove(request_id)
        self.running_req_ids.add(request_id)

        # Update first waiting idx
        self.first_waiting_idx = None
        for i in range(first_waiting_idx, len(self.combined)):
            if self.combined[i].request_id in self.waiting_req_ids:
                self.first_waiting_idx = i
                break

        # Update last running idx if necessary
        if self.first_waiting_idx is None and len(self.running_req_ids) > 0:
            self.last_running_idx = len(self.combined) - 1

        return

    def sync_idxs(self):
        first_waiting_idx = self.first_waiting_idx or len(self.combined) - 1

        self.first_waiting_idx = None
        for i in range(first_waiting_idx, -1, -1):
            if self.combined[i].request_id not in self.waiting_req_ids:
                self.first_waiting_idx = i + 1
                break

        if self.last_running_idx is None:
            return

        last_running_idx = self.last_running_idx

        self.last_running_idx = None
        for i in range(last_running_idx, len(self.combined)):
            if self.combined[i].request_id not in self.running_req_ids:
                self.last_running_idx = i - 1
                break
