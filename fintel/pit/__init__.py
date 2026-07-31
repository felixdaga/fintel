"""Point-in-time. The guarantee no strategy can be trusted to enforce on itself.

Everything a data source returns passes through `Cutoff` before an agent sees it.
"""

from fintel.pit.clamp import Cutoff

__all__ = ["Cutoff"]
