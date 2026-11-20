"""OwlBank TAC bridge: TAC v2.5.0 (Orchestrator mode) <-> a Strands agent loop on Bedrock.

Modules:

* :mod:`tac_bridge.agent_client`: the ``AgentClient`` protocol, its events and errors.
* :mod:`tac_bridge.agent_client_strands`: ``StrandsAgentClient``, the agent loop.
* :mod:`tac_bridge.bridge`: the per-call loop (prompt -> invoke -> stream -> inline tool / handoff).
* :mod:`tac_bridge.app`: our FastAPI app with TAC's ``/ws`` and ``/webhook`` handlers only.

Importing the package makes no network calls; ``app.build_app_from_env`` does
(``TAC(...)`` fetches the Conversation Configuration).
"""
