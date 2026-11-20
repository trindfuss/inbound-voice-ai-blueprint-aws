"""Every Lambda's entry point must exist under the exact name CDK configures.

The first real deploy found `Runtime.HandlerNotFound` on the front door: CDK pointed at
`lambdas.frontdoor.handler.handler`, but the module only defined `lambda_handler`. The unit tests
called the function directly, so nothing tied the two together. This test reads the handler string
CDK uses and imports each module the way the Lambda runtime does.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

COMPUTE_STACK = Path(__file__).resolve().parents[2] / "infra" / "owlbank_infra" / "compute_stack.py"
FUNCTIONS = ("frontdoor", "auth_service", "bank_tools", "level0_tools", "context_api", "flex_authorizer",
             "recap_writer")


def _cdk_handler_template() -> str:
    m = re.search(r'handler=f"(lambdas\.\{name\}\.([a-z_]+)\.([a-z_]+))"', COMPUTE_STACK.read_text())
    assert m, "could not find the Lambda handler string in compute_stack.py"
    return m.group(1)


def test_cdk_handler_string_shape() -> None:
    assert _cdk_handler_template() == "lambdas.{name}.handler.handler"


@pytest.mark.parametrize("name", FUNCTIONS)
def test_entry_point_exists_where_cdk_points(name: str) -> None:
    dotted = _cdk_handler_template().format(name=name)
    module_path, _, attr = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    fn = getattr(module, attr, None)
    assert callable(fn), f"{dotted} is missing: the Lambda runtime would fail with HandlerNotFound"
