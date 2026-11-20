import assert from "node:assert/strict";
import { test } from "node:test";

import { ApiError, apiUrl, createClient, mapError } from "../lib/api.js";
import { parseRoute } from "../lib/route.js";

test("error mapping: 400 names the field", () => {
  const e = mapError(400, { error: "invalid", message: "must be an integer from 0 to 10000", field: "tools.get_balance.mask.after_ms" });
  assert.equal(e.kind, "invalid");
  assert.equal(e.field, "tools.get_balance.mask.after_ms");
  assert.equal(e.message, "tools.get_balance.mask.after_ms: must be an integer from 0 to 10000");
});

test("error mapping: 409 with current is a conflict; 409 without is a refusal", () => {
  const c = mapError(409, { error: "conflict", message: "the tool changed since it was read", current: { rev: 4 } });
  assert.equal(c.kind, "conflict");
  assert.deepEqual(c.current, { rev: 4 });
  assert.equal(mapError(409, { error: "last_admin", message: "keep one admin" }).kind, "invalid");
});

test("error mapping: 401, 403, 404, 429, 503, 5xx, network", () => {
  assert.equal(mapError(401, { message: "Unauthorized" }).kind, "auth");
  const f = mapError(403, { error: "forbidden", message: "your group can't do this" });
  assert.equal(f.kind, "forbidden");
  assert.equal(f.message, "your group can't do this");
  assert.match(mapError(403, null).message, /group/);
  assert.equal(mapError(403, { error: "self_delete", message: "x" }).message, "You can't delete your own user.");
  assert.equal(mapError(404, {}).kind, "not_found");
  assert.equal(mapError(429, {}).kind, "throttled");
  assert.equal(mapError(503, { error: "not_seeded" }).kind, "unavailable");
  assert.equal(mapError(502, "not json").kind, "server");
  assert.equal(mapError(0, null).kind, "network");
});

test("error mapping property: every status maps to a kind with a title and a message", () => {
  for (let s = 0; s < 600; s++) {
    const e = mapError(s, s % 2 ? { error: "x" } : null);
    assert.ok(e.kind && e.title && e.message, String(s));
  }
});

test("URLs encode each path segment (usernames are emails) and drop empty query values", () => {
  assert.equal(apiUrl("/api", ["users", "ops+1@example.com", "resend"]), "/api/users/ops%2B1%40example.com/resend");
  assert.equal(apiUrl("/api/", ["prompts"], { prefix: "mask.", x: "" }), "/api/prompts?prefix=mask.");
  assert.equal(apiUrl("/api", ["audit"], { item: "policy#tool#get_balance", limit: 100 }),
    "/api/audit?item=policy%23tool%23get_balance&limit=100");
  assert.equal(apiUrl("/api", ["tools"]), "/api/tools");
});

function fakeFetch(responses) {
  const calls = [];
  const impl = async (url, init) => {
    calls.push({ url, init });
    const [status, body] = responses.shift();
    return { ok: status >= 200 && status < 300, status, text: async () => (body === undefined ? "" : JSON.stringify(body)) };
  };
  return { impl, calls };
}

test("client sends the token and JSON, same origin, no cookies", async () => {
  const f = fakeFetch([[200, { changed: true }]]);
  const api = createClient({ base: "/api", getToken: async () => "tok", fetchImpl: f.impl });
  assert.deepEqual(await api.put(["tools", "get_balance"], { rev: 1, min_level: 2 }), { changed: true });
  const { url, init } = f.calls[0];
  assert.equal(url, "/api/tools/get_balance");
  assert.equal(init.method, "PUT");
  assert.equal(init.headers.Authorization, "tok");
  assert.equal(init.headers["Content-Type"], "application/json");
  assert.equal(init.credentials, "omit");
  assert.deepEqual(JSON.parse(init.body), { rev: 1, min_level: 2 });
});

test("client: a 401 refreshes once and retries; a second 401 signs out", async () => {
  const f = fakeFetch([[401, { message: "Unauthorized" }], [200, { ok: 1 }]]);
  let refreshed = 0;
  const api = createClient({ base: "/api", getToken: async () => "t", refresh: async () => ++refreshed > 0, fetchImpl: f.impl });
  assert.deepEqual(await api.get(["tools"]), { ok: 1 });
  assert.equal(refreshed, 1);

  const g = fakeFetch([[401, {}], [401, {}]]);
  let lost = 0;
  const api2 = createClient({ base: "/api", getToken: async () => "t", refresh: async () => true, onAuthLost: () => lost++, fetchImpl: g.impl });
  await assert.rejects(api2.get(["tools"]), (e) => e instanceof ApiError && e.kind === "auth");
  assert.equal(lost, 1);
  assert.equal(g.calls.length, 2);
});

test("client: errors carry the mapped fields; network failures are kind network", async () => {
  const f = fakeFetch([[409, { error: "conflict", message: "changed", current: { rev: 3 } }]]);
  const api = createClient({ base: "/api", getToken: async () => "t", fetchImpl: f.impl });
  await assert.rejects(api.put(["masking"], { rev: 2, masking: {} }), (e) => e.kind === "conflict" && e.current.rev === 3);
  const down = createClient({ base: "/api", getToken: async () => "t", fetchImpl: async () => { throw new TypeError("offline"); } });
  await assert.rejects(down.get(["tools"]), (e) => e.kind === "network");
});

test("routes", () => {
  assert.equal(parseRoute("").view, "tools");
  assert.equal(parseRoute("#/nope").view, "tools");
  assert.deepEqual([parseRoute("#/tools/get_balance").view, parseRoute("#/tools/get_balance").arg], ["tools", "get_balance"]);
  const a = parseRoute("#/audit?item=policy%23tool%23get_balance");
  assert.equal(a.view, "audit");
  assert.equal(a.params.get("item"), "policy#tool#get_balance");
  assert.equal(parseRoute("#/prompts?tool=get_balance").params.get("tool"), "get_balance");
  assert.equal(parseRoute("#/users").view, "users");
});
