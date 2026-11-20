"""WAF web ACLs (REGIONAL for API Gateway and the public ALB, CLOUDFRONT for the CloudFront
mode): AWS managed rules, a rate-based rule, a body size limit, and an allow-list of paths
and methods."""

from __future__ import annotations

from collections.abc import Sequence

from aws_cdk import aws_wafv2 as waf
from constructs import Construct

W = waf.CfnWebACL

#: Label the Common Rule Set adds when ``SizeRestrictions_BODY`` (body over 8 KB) matches, whatever
#: that rule's action is. The path-scoped size rule below keys on it.
CRS_SIZE_LABEL = "awswaf:managed:aws:core-rule-set:SizeRestrictions_Body"
CRS_NAME = "AWSManagedRulesCommonRuleSet"
#: CloudFront's WAF body inspection limit can be raised to 64 KB (default 16 KB); the settings are
#: KB_16, KB_32, KB_48 and KB_64. A body past the limit is "oversize" (BodySizeLimit's MATCH).
#: A REGIONAL ACL can raise it the same way for API Gateway (``inspection_resource="API_GATEWAY"``).
CLOUDFRONT_MAX_INSPECTION_BYTES = 65536
MAX_INSPECTION_BYTES = CLOUDFRONT_MAX_INSPECTION_BYTES
#: Regional resource types whose body inspection limit an ACL's ``AssociationConfig`` can raise.
REGIONAL_INSPECTION_RESOURCES = ("API_GATEWAY",)
MANAGED_MODES = ("count", "block")
#: The HTTP Basic challenge a request without credentials gets (the front door sends the same).
BASIC_CHALLENGE_STATUS = 401
BASIC_CHALLENGE_HEADER = 'Basic realm="owlbank"'


def _managed(name: str, priority: int, mode: str, *, size_body_to_count: bool = False) -> W.RuleProperty:
    """A managed rule group. ``count`` mode: the group's override action is Count (it still evaluates
    and logs, nothing it matches is blocked). ``block`` mode: the group's own actions apply, except
    ``SizeRestrictions_BODY`` set to Count when ``size_body_to_count`` (see ``web_acl``)."""
    overrides = ([W.RuleActionOverrideProperty(name="SizeRestrictions_BODY",
                                               action_to_use=W.RuleActionProperty(count={}))]
                 if size_body_to_count and mode == "block" else None)
    return W.RuleProperty(
        name=name, priority=priority,
        override_action=W.OverrideActionProperty(count={}) if mode == "count"
        else W.OverrideActionProperty(none={}),
        statement=W.StatementProperty(managed_rule_group_statement=W.ManagedRuleGroupStatementProperty(
            vendor_name="AWS", name=name, rule_action_overrides=overrides)),
        visibility_config=_vis(name),
    )


def _vis(name: str) -> W.VisibilityConfigProperty:
    return W.VisibilityConfigProperty(cloud_watch_metrics_enabled=True, metric_name=name,
                                      sampled_requests_enabled=True)


def _path_method(path_regex: str, methods: list[str]) -> W.StatementProperty:
    path = W.StatementProperty(regex_match_statement=W.RegexMatchStatementProperty(
        regex_string=path_regex, field_to_match=W.FieldToMatchProperty(uri_path={}),
        text_transformations=[W.TextTransformationProperty(priority=0, type="NONE")]))
    method_stmts = [W.StatementProperty(byte_match_statement=W.ByteMatchStatementProperty(
        search_string=m, positional_constraint="EXACTLY",
        field_to_match=W.FieldToMatchProperty(method={}),
        text_transformations=[W.TextTransformationProperty(priority=0, type="NONE")])) for m in methods]
    method = method_stmts[0] if len(method_stmts) == 1 else W.StatementProperty(
        or_statement=W.OrStatementProperty(statements=method_stmts))
    return W.StatementProperty(and_statement=W.AndStatementProperty(statements=[path, method]))


def _uri_regex(regex: str) -> W.StatementProperty:
    return W.StatementProperty(regex_match_statement=W.RegexMatchStatementProperty(
        regex_string=regex, field_to_match=W.FieldToMatchProperty(uri_path={}),
        text_transformations=[W.TextTransformationProperty(priority=0, type="NONE")]))


def _body_over(size: int) -> W.StatementProperty:
    """The body is larger than ``size`` bytes, or larger than WAF inspects (oversize matches)."""
    return W.StatementProperty(size_constraint_statement=W.SizeConstraintStatementProperty(
        comparison_operator="GT", size=size,
        field_to_match=W.FieldToMatchProperty(body=W.BodyProperty(oversize_handling="MATCH")),
        text_transformations=[W.TextTransformationProperty(priority=0, type="NONE")]))


def _outside(statement: W.StatementProperty, paths: W.StatementProperty | None) -> W.StatementProperty:
    """``statement`` and not on ``paths``."""
    if paths is None:
        raise ValueError("no paths to exclude")
    return W.StatementProperty(and_statement=W.AndStatementProperty(statements=[
        statement, W.StatementProperty(not_statement=W.NotStatementProperty(statement=paths))]))


def _inspection_limit(max_body_bytes: int) -> str:
    """Smallest body inspection setting (CloudFront, or a raised regional one) that covers
    ``max_body_bytes``."""
    if max_body_bytes > MAX_INSPECTION_BYTES:
        raise ValueError(f"WAF inspects at most {MAX_INSPECTION_BYTES} bytes of a body")
    return f"KB_{max(1, -(-max_body_bytes // 16384)) * 16}"


def web_acl(scope: Construct, cid: str, *, name: str, metric: str,
            allowed: list[tuple[str, list[str]]], rate_limit: int, max_body_bytes: int,
            scope_type: str = "REGIONAL", managed_mode: str = "block",
            large_body_paths: Sequence[str] = (), authorization_required_path: str | None = None,
            required_header: tuple[str, str] | None = None,
            large_body_max_bytes: int | None = None, inspection_resource: str | None = None,
            own_rate_paths: Sequence[str] = ()) -> W:
    """``allowed`` is a list of (URI path regex, methods); every other request is blocked.

    ``scope_type`` is ``REGIONAL`` or ``CLOUDFRONT`` (the latter only deploys in us-east-1).

    ``managed_mode`` (``wafManagedRuleMode``): ``count`` runs the three AWS managed rule groups in
    Count (observe the WAF logs first), ``block`` enforces them.

    **Body size.** ``BodySizeLimit`` (``max_body_bytes``, oversize bodies match) is the hard cap.
    Two other limits sit under it and would silently undercut it: the host's inspection limit (a
    CloudFront ACL's is set to the highest value that covers ``max_body_bytes``, 64 KB at most; the
    API Gateway default of 16 KB covers the API's 8 KB cap; an ALB's is fixed at 8 KB) and the Common
    Rule Set's ``SizeRestrictions_BODY`` (8 KB). For paths in ``large_body_paths`` (URI regexes that
    legitimately send more than 8 KB), ``block`` mode sets that one rule to Count and a custom rule
    on its label blocks oversize bodies on every *other* path, so the CRS keeps its 8 KB limit
    everywhere else. In ``count`` mode that custom rule counts too.

    ``authorization_required_path``: a URI regex whose requests are blocked when they carry no
    ``Authorization`` header. The block is a custom response, ``401`` with
    ``WWW-Authenticate: Basic realm="owlbank"`` and no body: Twilio sends the first request of a
    password-protected URL *without* credentials and repeats it with ``Authorization: Basic`` only
    after such a challenge, so a plain 403 would end every call.

    ``large_body_max_bytes``: a second, larger hard cap for ``large_body_paths`` only. ``BodySizeLimit``
    (``max_body_bytes``) then applies to every *other* path, whatever ``managed_mode`` is, and
    ``LargeBodySizeLimit`` blocks bodies over ``large_body_max_bytes`` on the large-body paths. Without
    it, ``max_body_bytes`` is the one cap for every path (as before).

    ``inspection_resource``: for a ``REGIONAL`` ACL, the resource type (``API_GATEWAY``) whose body
    inspection limit is raised to cover the largest cap (default 16 KB), as a CloudFront ACL's is.

    ``own_rate_paths``: URI regexes whose requests get their own per-IP rate count
    (``RateLimitPerIpOwnPaths``, the same ``rate_limit``) and are left out of ``RateLimitPerIp``'s, so a
    high-volume callback on one path cannot use up the count of the others from the same IP (human-leg
    ``/voice/transcription``). A rate-based rule counts per instance and per aggregation key (the
    IP), and with a scope-down statement only the requests that match it (AWS WAF developer guide,
    rate-based rule statements; API ``RateBasedStatement``). It runs right after ``RateLimitPerIp``
    (priority 2), before the body and header rules, as the per-IP count does.

    ``required_header``: ``(name, value)``; every request whose header ``name`` is not exactly
    ``value`` is blocked (403), whatever ``managed_mode`` is. The admin API uses it for the
    ``X-Origin-Verify`` value only its CloudFront distribution sends; ``value`` is then a
    Secrets Manager dynamic reference, so the template holds no secret.
    """
    if managed_mode not in MANAGED_MODES:
        raise ValueError(f"managed_mode must be one of {MANAGED_MODES}")
    if large_body_max_bytes is not None and (not large_body_paths or large_body_max_bytes <= max_body_bytes):
        raise ValueError("large_body_max_bytes needs large_body_paths and must exceed max_body_bytes")
    if inspection_resource is not None and (scope_type != "REGIONAL"
                                            or inspection_resource not in REGIONAL_INSPECTION_RESOURCES):
        raise ValueError(f"inspection_resource must be one of {REGIONAL_INSPECTION_RESOURCES} (REGIONAL only)")
    big = [_uri_regex(p) for p in large_body_paths]
    on_big_path = (None if not big else big[0] if len(big) == 1
                   else W.StatementProperty(or_statement=W.OrStatementProperty(statements=big)))
    own = [_uri_regex(p) for p in own_rate_paths]
    on_own_path = (None if not own else own[0] if len(own) == 1
                   else W.StatementProperty(or_statement=W.OrStatementProperty(statements=own)))
    allow_stmts = [_path_method(p, m) for p, m in allowed]
    allow_any = allow_stmts[0] if len(allow_stmts) == 1 else W.StatementProperty(
        or_statement=W.OrStatementProperty(statements=allow_stmts))
    rules = [
        W.RuleProperty(
            name="AllowedPathsAndMethodsOnly", priority=0,
            action=W.RuleActionProperty(block={}),
            statement=W.StatementProperty(not_statement=W.NotStatementProperty(statement=allow_any)),
            visibility_config=_vis(f"{metric}-paths")),
        W.RuleProperty(
            name="RateLimitPerIp", priority=1,
            action=W.RuleActionProperty(block={}),
            statement=W.StatementProperty(rate_based_statement=W.RateBasedStatementProperty(
                aggregate_key_type="IP", limit=rate_limit,
                scope_down_statement=None if on_own_path is None else W.StatementProperty(
                    not_statement=W.NotStatementProperty(statement=on_own_path)))),
            visibility_config=_vis(f"{metric}-rate")),
        W.RuleProperty(
            name="BodySizeLimit", priority=3,
            action=W.RuleActionProperty(block={}),
            statement=_outside(_body_over(max_body_bytes), on_big_path) if large_body_max_bytes
            else _body_over(max_body_bytes),
            visibility_config=_vis(f"{metric}-size")),
        _managed("AWSManagedRulesAmazonIpReputationList", 10, managed_mode),
        _managed(CRS_NAME, 11, managed_mode, size_body_to_count=bool(large_body_paths)),
        _managed("AWSManagedRulesKnownBadInputsRuleSet", 13, managed_mode),
    ]
    if authorization_required_path:
        rules.append(W.RuleProperty(
            name="AuthorizationHeaderRequired", priority=4,
            action=W.RuleActionProperty(block=W.BlockActionProperty(
                custom_response=W.CustomResponseProperty(
                    response_code=BASIC_CHALLENGE_STATUS,
                    response_headers=[W.CustomHTTPHeaderProperty(
                        name="WWW-Authenticate", value=BASIC_CHALLENGE_HEADER)]))),
            statement=W.StatementProperty(and_statement=W.AndStatementProperty(statements=[
                _uri_regex(authorization_required_path),
                W.StatementProperty(not_statement=W.NotStatementProperty(
                    statement=W.StatementProperty(size_constraint_statement=W.SizeConstraintStatementProperty(
                        comparison_operator="GT", size=0,
                        field_to_match=W.FieldToMatchProperty(single_header={"Name": "authorization"}),
                        text_transformations=[W.TextTransformationProperty(priority=0, type="NONE")]))))])),
            visibility_config=_vis(f"{metric}-noauth")))
    if required_header:
        header, value = required_header
        rules.append(W.RuleProperty(
            name="OriginVerifyHeaderRequired", priority=5,
            action=W.RuleActionProperty(block={}),
            statement=W.StatementProperty(not_statement=W.NotStatementProperty(
                statement=W.StatementProperty(byte_match_statement=W.ByteMatchStatementProperty(
                    search_string=value, positional_constraint="EXACTLY",
                    field_to_match=W.FieldToMatchProperty(single_header={"Name": header.lower()}),
                    text_transformations=[W.TextTransformationProperty(priority=0, type="NONE")])))),
            visibility_config=_vis(f"{metric}-origin-verify")))
    if on_own_path is not None:
        rules.append(W.RuleProperty(
            name="RateLimitPerIpOwnPaths", priority=2,
            action=W.RuleActionProperty(block={}),
            statement=W.StatementProperty(rate_based_statement=W.RateBasedStatementProperty(
                aggregate_key_type="IP", limit=rate_limit, scope_down_statement=on_own_path)),
            visibility_config=_vis(f"{metric}-rate-own")))
    if large_body_max_bytes and on_big_path is not None:
        rules.append(W.RuleProperty(
            name="LargeBodySizeLimit", priority=6,
            action=W.RuleActionProperty(block={}),
            statement=W.StatementProperty(and_statement=W.AndStatementProperty(statements=[
                _body_over(large_body_max_bytes), on_big_path])),
            visibility_config=_vis(f"{metric}-size-large")))
    if on_big_path is not None:
        rules.append(W.RuleProperty(
            name="OversizeBodyOutsideLargeBodyPaths", priority=12,
            action=W.RuleActionProperty(count={}) if managed_mode == "count" else W.RuleActionProperty(block={}),
            statement=W.StatementProperty(and_statement=W.AndStatementProperty(statements=[
                W.StatementProperty(label_match_statement=W.LabelMatchStatementProperty(
                    scope="LABEL", key=CRS_SIZE_LABEL)),
                W.StatementProperty(not_statement=W.NotStatementProperty(statement=on_big_path))])),
            visibility_config=_vis(f"{metric}-oversize")))
    inspected = "CLOUDFRONT" if scope_type == "CLOUDFRONT" else inspection_resource
    association = (W.AssociationConfigProperty(request_body={
        inspected: W.RequestBodyAssociatedResourceTypeConfigProperty(
            default_size_inspection_limit=_inspection_limit(max(max_body_bytes, large_body_max_bytes or 0)))})
        if inspected else None)
    return W(scope, cid, name=name, scope=scope_type, default_action=W.DefaultActionProperty(allow={}),
             rules=sorted(rules, key=lambda r: r.priority), visibility_config=_vis(metric),
             association_config=association)
