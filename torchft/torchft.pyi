from datetime import timedelta
from typing import Optional, Tuple

class ManagerClient:
    def __init__(self, addr: str, timeout: timedelta) -> None: ...
    def quorum(
        self, rank: int, step: int, checkpoint_server_addr: str
    ) -> Tuple[int, int, int, str, str, int, Optional[int], int, bool]: ...
    def checkpoint_address(self, rank: int) -> str: ...
    def should_commit(self, rank: int, step: int, should_commit: bool) -> bool: ...