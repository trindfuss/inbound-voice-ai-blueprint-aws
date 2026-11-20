"""Call observability: OpenTelemetry for the bridge and the front door (``callObservability``).

The package is shared by both processes:

* :mod:`~owlbank.telemetry.settings`: the switch and its settings from the environment
  (stdlib only; it never imports OpenTelemetry);
* :mod:`~owlbank.telemetry.allowlist`: the span attribute and event keys that may leave the
  process, each with its reason;
* :mod:`~owlbank.telemetry.filtering`: the content filter, an exporter wrapper;
* :mod:`~owlbank.telemetry.evalshape`: the span shape AgentCore Evaluations reads;
* :mod:`~owlbank.telemetry.sampling`: per-call sampling from the CallSid;
* :mod:`~owlbank.telemetry.health`: export failures counted and logged once a minute.

Every module except ``settings`` and ``allowlist`` imports OpenTelemetry, so this ``__init__``
imports none of them: a process imports them only after :meth:`TelemetrySettings.from_env`
says the switch is on.
"""

from owlbank.telemetry.settings import TelemetrySettings

__all__ = ["TelemetrySettings"]
