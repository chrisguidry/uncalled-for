from __future__ import annotations

import sys

# A ``type`` statement is a syntax error before Python 3.12, so the module
# that uses them cannot even be collected there.
collect_ignore = (
    [] if sys.version_info >= (3, 12) else ["test_annotated_type_aliases.py"]
)
