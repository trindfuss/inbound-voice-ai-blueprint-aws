"""Make a Node.js binary available to jsii before ``aws_cdk`` is imported.

aws-cdk-lib in Python runs on jsii, which starts a ``node`` child process. To keep
synth self-contained (no system Node, no npm CDK CLI), the ``nodejs-wheel-binaries``
dependency ships ``node`` inside this project's venv, pinned to a release jsii supports.
That one is put first on ``PATH``, so synth doesn't depend on the machine's Node (a newer
system Node makes jsii warn that it is untested). A ``node`` already on ``PATH`` is used
only if the wheel is missing.
"""

from __future__ import annotations

import os


def ensure_node() -> None:
    try:
        import nodejs_wheel  # from nodejs-wheel-binaries
    except ImportError:  # pragma: no cover - dependency missing
        return
    bin_dir = os.path.join(os.path.dirname(nodejs_wheel.__file__), "bin")
    path = os.environ.get("PATH", "")
    if path.split(os.pathsep)[0] != bin_dir:
        os.environ["PATH"] = bin_dir + os.pathsep + path


ensure_node()
