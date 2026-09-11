"""What this process cost to get here — for the cold-start measurement.

`scripts/cold_start.sh` times requests from outside, which is the number a
visitor experiences, and the only one the acceptance criterion is applied to.
But a single outside number cannot say *where* the seconds went: platform
wake, interpreter and imports, unpickling the artifact, or the request itself.
So `/health` reports three facts the process can measure about itself, and the
script prints them beside the outside timing, so that a slow cold start on the
day arrives with its decomposition rather than as one opaque figure.

None of this is measured unless asked. `process_age_seconds` reads `/proc`
where there is one — the true age of the process, which includes the
interpreter start and the imports that happen before this module is reached —
and otherwise falls back to the age of this module, which is imported before
the heavy ones and so misses only the interpreter's own start.
"""

from __future__ import annotations

import os
import resource
import sys
import time

#: Taken as early as the import order allows; `api.main` imports this module
#: before anything that pulls in scikit-learn or XGBoost.
_IMPORTED_AT = time.monotonic()


def process_age_seconds() -> float:
    """Seconds since this process started, as well as the platform can say."""
    try:
        with open("/proc/self/stat") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        with open("/proc/uptime") as handle:
            uptime = float(handle.read().split()[0])
        # Field 22 of /proc/[pid]/stat is starttime in clock ticks since boot;
        # after the `)` split it is at index 19.
        ticks = os.sysconf("SC_CLK_TCK")
        return uptime - int(fields[19]) / ticks
    except (OSError, IndexError, ValueError):
        return time.monotonic() - _IMPORTED_AT


def rss_mb() -> float:
    """Peak resident set size of this process, in MiB.

    `ru_maxrss` is kilobytes on Linux and bytes on macOS — the one place the
    two disagree, and the reason this is a function rather than a field read.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024.0 if sys.platform != "darwin" else peak / (1024.0 * 1024.0)
