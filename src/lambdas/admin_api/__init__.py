"""The admin console's API: one Lambda behind API Gateway.

* :mod:`.app`: routing, the group checks and the JSON responses (pure, injected deps).
* :mod:`.config_routes`: tools, masking, prompts, placeholders, audit and restore.
* :mod:`.users`: user management for admins over a Cognito client protocol.
* :mod:`.audit`: audit entries.
* :mod:`.handler`: the Lambda entry point and its environment.
"""
