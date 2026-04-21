"""Deterministic scenario catalog for the rebuilt SPECTRA benchmark."""

from __future__ import annotations

import copy
import random
import uuid
from dataclasses import dataclass
from typing import Any

from .causal_graph import DEFAULT_SERVICE_GRAPH
from .state import (
    AGENT_NAMES,
    AgentName,
    CommanderActionType,
    Difficulty,
    HealthStatus,
    MasterSREState,
    ScenarioObjective,
    SERVICE_NAMES,
    ServiceName,
)


@dataclass(frozen=True)
class ScenarioBlueprint:
    scenario_id: str
    name: str
    domain: str
    description: str
    difficulty: Difficulty
    step_budget: int
    root_cause_service: ServiceName
    root_cause_vector: str
    causal_chain: tuple[str, ...]
    common_trap: str
    decoy_agent: AgentName | None
    decoy_service: ServiceName | None
    best_followup_agent: AgentName | None
    required_plan: tuple[tuple[CommanderActionType, ServiceName | None], ...]
    specialist_signal_map: dict[AgentName, str]
    cpu_pct: dict[ServiceName, float]
    mem_pct: dict[ServiceName, float]
    latency_ms: dict[ServiceName, float]
    error_rate: dict[ServiceName, float]
    health: dict[ServiceName, HealthStatus]
    log_lines: list[str]
    error_types: dict[str, int]
    stack_traces: list[str]
    query_patterns: list[str]
    event_sequence: list[dict[str, Any]]
    alert_strings: list[str]
    auth_fail_count: int
    suspicious_ips: list[str]
    injection_patterns: list[str]
    cve_flags: list[str]
    followup_notes: dict[AgentName, str]


def _event(service: ServiceName, message: str, severity: str = "info") -> dict[str, Any]:
    return {"service": service, "message": message, "severity": severity}


SCENARIOS: dict[str, ScenarioBlueprint] = {
    "database_sqli_outage": ScenarioBlueprint(
        scenario_id="database_sqli_outage",
        name="Database SQLi Outage",
        domain="payments",
        description="Login SQL injection overloads the database and drags the checkout path down with it.",
        difficulty="easy",
        step_budget=8,
        root_cause_service="database",
        root_cause_vector="SQL injection on the login query builder",
        causal_chain=("database", "cache", "api-gateway", "worker"),
        common_trap="Restarting api-gateway only treats the visible symptom; the exploit path on database remains open.",
        decoy_agent=None,
        decoy_service=None,
        best_followup_agent="security",
        required_plan=(
            ("isolate_service", "database"),
            ("rollback_config", "database"),
            ("restart_service", "database"),
            ("submit_resolution", None),
        ),
        specialist_signal_map={
            "infra": "Database clearly looks worst but cannot explain why.",
            "log": "SQL error signatures strongly implicate database query handling.",
            "security": "Security alert names SQL injection directly.",
        },
        cpu_pct={"api-gateway": 61.0, "database": 94.0, "cache": 72.0, "worker": 45.0, "auth_service": 26.0},
        mem_pct={"api-gateway": 58.0, "database": 88.0, "cache": 63.0, "worker": 40.0, "auth_service": 28.0},
        latency_ms={"api-gateway": 420.0, "database": 1200.0, "cache": 380.0, "worker": 210.0, "auth_service": 120.0},
        error_rate={"api-gateway": 0.18, "database": 0.41, "cache": 0.22, "worker": 0.09, "auth_service": 0.03},
        health={"api-gateway": "degraded", "database": "critical", "cache": "degraded", "worker": "healthy", "auth_service": "healthy"},
        log_lines=[
            "ERROR: syntax error at or near \"'\" (SQL)",
            "CRITICAL: query rejected — injection detected",
            "WARN: database connection pool saturation on login flow",
        ],
        error_types={"SQLSyntaxError": 847, "QueryRejected": 312, "ConnectionTimeout": 89},
        stack_traces=["psycopg2.errors.SyntaxError: unterminated quoted string at or near \"'\""],
        query_patterns=["SELECT * FROM users WHERE id = '' OR '1'='1'", "DROP TABLE--"],
        event_sequence=[
            _event("database", "Database query parser overwhelmed by malformed login queries", "critical"),
            _event("api-gateway", "Gateway retries rising after database timeouts", "warning"),
        ],
        alert_strings=["SECURITY_ALERT: SQL_INJECTION detected on database service"],
        auth_fail_count=3,
        suspicious_ips=["192.168.1.105", "10.0.0.44"],
        injection_patterns=["SQL_INJECTION: OR '1'='1'", "SQL_INJECTION: DROP TABLE"],
        cve_flags=["CWE-89: Improper Neutralization of SQL Commands"],
        followup_notes={
            "infra": "Database memory is pinned because the login path keeps opening toxic queries.",
            "log": "The malformed query payload is coming from the login flow, not a downstream worker.",
            "security": "The alert maps to CWE-89 and the fastest safe containment is to isolate database write traffic.",
        },
    ),
    "api_gateway_xss": ScenarioBlueprint(
        scenario_id="api_gateway_xss",
        name="API Gateway XSS Spill",
        domain="customer-facing web",
        description="Reflected XSS in the gateway poisons responses and degrades downstream traffic handling.",
        difficulty="medium",
        step_budget=8,
        root_cause_service="api-gateway",
        root_cause_vector="Cross-site scripting in response rendering",
        causal_chain=("api-gateway", "database", "cache"),
        common_trap="Scaling database helps the symptom for a moment, but the exploit is still entering through api-gateway.",
        decoy_agent=None,
        decoy_service=None,
        best_followup_agent="log",
        required_plan=(
            ("isolate_service", "api-gateway"),
            ("rollback_config", "api-gateway"),
            ("scale_service", "api-gateway"),
            ("submit_resolution", None),
        ),
        specialist_signal_map={
            "infra": "Gateway is clearly the bottleneck.",
            "log": "Reflected payloads and CSP violations identify the application path directly.",
            "security": "Security has the right class of attack but weaker operational context.",
        },
        cpu_pct={"api-gateway": 88.0, "database": 71.0, "cache": 58.0, "worker": 39.0, "auth_service": 33.0},
        mem_pct={"api-gateway": 76.0, "database": 62.0, "cache": 49.0, "worker": 35.0, "auth_service": 30.0},
        latency_ms={"api-gateway": 640.0, "database": 420.0, "cache": 280.0, "worker": 190.0, "auth_service": 160.0},
        error_rate={"api-gateway": 0.34, "database": 0.19, "cache": 0.14, "worker": 0.06, "auth_service": 0.04},
        health={"api-gateway": "critical", "database": "degraded", "cache": "degraded", "worker": "healthy", "auth_service": "healthy"},
        log_lines=[
            "ERROR: XSS payload detected in request param 'q'",
            "WARN: Content-Security-Policy violation on reflected response",
            "ERROR: InvalidRequestParam in api-gateway templating layer",
        ],
        error_types={"XSSPayloadDetected": 612, "CSPViolation": 289, "InvalidRequestParam": 445},
        stack_traces=["TemplateRenderError: reflected script tag escaped policy boundary"],
        query_patterns=["<script>alert(1)</script>", "javascript:void(0)", "onerror=alert(document.cookie)"],
        event_sequence=[
            _event("api-gateway", "Gateway rendering path hit by repeated malicious script payloads", "critical"),
            _event("database", "Database load elevated after gateway retry storm", "warning"),
        ],
        alert_strings=["SECURITY_ALERT: XSS_ATTEMPT detected on api-gateway"],
        auth_fail_count=12,
        suspicious_ips=["172.16.0.51", "172.16.0.52"],
        injection_patterns=["XSS: script_tag_injection", "XSS: event_handler_injection"],
        cve_flags=["CWE-79: Cross-site Scripting"],
        followup_notes={
            "infra": "Gateway saturation is the origin problem, not just a downstream retry effect.",
            "log": "The rendering template changed in the last deployment and lines up with the first malicious payloads.",
            "security": "The XSS signature is real, but the logs are better for pinpointing the exact rollback target.",
        },
    ),
    "broken_auth_cascade": ScenarioBlueprint(
        scenario_id="broken_auth_cascade",
        name="Broken Auth Cascade",
        domain="identity",
        description="JWT forgery on auth_service floods cache lookups and creates a misleading metrics profile.",
        difficulty="hard",
        step_budget=7,
        root_cause_service="auth_service",
        root_cause_vector="Broken authentication / JWT forgery",
        causal_chain=("auth_service", "cache", "api-gateway", "database"),
        common_trap="Cache looks like the hottest service, but it is only the victim of the auth storm.",
        decoy_agent="infra",
        decoy_service="cache",
        best_followup_agent="security",
        required_plan=(
            ("request_followup", None),
            ("isolate_service", "auth_service"),
            ("rollback_config", "auth_service"),
            ("restart_service", "auth_service"),
            ("submit_resolution", None),
        ),
        specialist_signal_map={
            "infra": "Infra sees cache as the most degraded service and is intentionally misleading here.",
            "log": "JWT verification failures point back to auth_service.",
            "security": "Security has the strongest and most specific signal with a named CVE.",
        },
        cpu_pct={"api-gateway": 78.0, "database": 69.0, "cache": 94.0, "worker": 42.0, "auth_service": 91.0},
        mem_pct={"api-gateway": 70.0, "database": 61.0, "cache": 89.0, "worker": 36.0, "auth_service": 84.0},
        latency_ms={"api-gateway": 610.0, "database": 490.0, "cache": 920.0, "worker": 200.0, "auth_service": 890.0},
        error_rate={"api-gateway": 0.31, "database": 0.24, "cache": 0.51, "worker": 0.08, "auth_service": 0.62},
        health={"api-gateway": "degraded", "database": "degraded", "cache": "critical", "worker": "healthy", "auth_service": "critical"},
        log_lines=[
            "CRITICAL: JWT signature verification failed (1,203 times in 60s)",
            "ERROR: auth_token_invalid — forged token detected",
            "WARN: cache.get() timeout — auth validation queue overflow",
        ],
        error_types={"JWTVerificationFailed": 1203, "AuthTokenInvalid": 847, "CacheTimeout": 412},
        stack_traces=["jwt.exceptions.InvalidSignatureError: Signature verification failed"],
        query_patterns=["JWT_FORGE: alg:none bypass", "JWT_FORGE: weak_secret_brute"],
        event_sequence=[
            _event("auth_service", "Token validation storm detected", "critical"),
            _event("cache", "Cache lookup flood saturating auth validation path", "critical"),
            _event("api-gateway", "Authentication retries increasing upstream latency", "warning"),
        ],
        alert_strings=[
            "SECURITY_ALERT: BROKEN_AUTH — JWT_FORGERY detected on auth_service",
            "SECURITY_ALERT: CREDENTIAL_STUFFING — 47 source IPs in 60s",
        ],
        auth_fail_count=1203,
        suspicious_ips=[f"10.1.0.{index}" for index in range(10, 57)],
        injection_patterns=["JWT_FORGE: alg:none bypass", "JWT_FORGE: weak_secret_brute"],
        cve_flags=["CVE-2023-45812: JWT Algorithm Confusion", "CWE-287: Improper Authentication"],
        followup_notes={
            "infra": "Cache will cool down if auth_service is contained, but infra alone cannot prove that.",
            "log": "The first failure happened in auth_service before cache saturation appeared.",
            "security": "The named CVE maps to auth_service, not cache; ask for containment on auth_service first.",
        },
    ),
    "worker_supply_chain_compromise": ScenarioBlueprint(
        scenario_id="worker_supply_chain_compromise",
        name="Worker Supply Chain Compromise",
        domain="background jobs",
        description="A poisoned worker release keeps pushing malformed jobs that destabilize database and gateway.",
        difficulty="hard",
        step_budget=8,
        root_cause_service="worker",
        root_cause_vector="Supply-chain compromise in worker deployment",
        causal_chain=("worker", "database", "api-gateway"),
        common_trap="Database and gateway look noisy, but the corrupted build artifact originates from worker.",
        decoy_agent="infra",
        decoy_service="database",
        best_followup_agent="log",
        required_plan=(
            ("request_followup", None),
            ("isolate_service", "worker"),
            ("rollback_config", "worker"),
            ("restart_service", "worker"),
            ("submit_resolution", None),
        ),
        specialist_signal_map={
            "infra": "Infra points to database because it is absorbing the poisoned workload.",
            "log": "Worker retry loops reveal the origin release.",
            "security": "Package signature and provenance flags confirm a compromised worker image.",
        },
        cpu_pct={"api-gateway": 67.0, "database": 90.0, "cache": 34.0, "worker": 86.0, "auth_service": 25.0},
        mem_pct={"api-gateway": 61.0, "database": 83.0, "cache": 31.0, "worker": 78.0, "auth_service": 22.0},
        latency_ms={"api-gateway": 560.0, "database": 780.0, "cache": 90.0, "worker": 640.0, "auth_service": 110.0},
        error_rate={"api-gateway": 0.24, "database": 0.33, "cache": 0.04, "worker": 0.47, "auth_service": 0.02},
        health={"api-gateway": "degraded", "database": "critical", "cache": "healthy", "worker": "critical", "auth_service": "healthy"},
        log_lines=[
            "ERROR: worker job payload checksum mismatch",
            "CRITICAL: job retry loop emitting malformed SQL payloads",
            "WARN: downstream database constraint failures sourced from worker batch runner",
        ],
        error_types={"ChecksumMismatch": 401, "RetryLoop": 720, "ConstraintViolation": 388},
        stack_traces=["ValueError: signed package digest mismatch in worker release bundle"],
        query_patterns=["INSERT malformed_batch", "UPDATE poisoned_payload"],
        event_sequence=[
            _event("worker", "Compromised worker package started replaying malformed jobs", "critical"),
            _event("database", "Database choking on malformed batch writes from worker", "critical"),
            _event("api-gateway", "Gateway degraded by stale reads after worker poison", "warning"),
        ],
        alert_strings=[
            "SECURITY_ALERT: UNSIGNED_PACKAGE detected in worker deployment",
            "SECURITY_ALERT: ARTIFACT_PROVENANCE mismatch on worker image",
        ],
        auth_fail_count=0,
        suspicious_ips=["artifact-registry", "worker-build-bot"],
        injection_patterns=["SUPPLY_CHAIN: unsigned_artifact", "SUPPLY_CHAIN: provenance_gap"],
        cve_flags=["SLSA-L3 provenance violation"],
        followup_notes={
            "infra": "Database pressure is secondary; worker keeps refilling the queue with bad jobs.",
            "log": "The first bad batch appears immediately after the latest worker release SHA.",
            "security": "The image provenance failure is on worker, so rollback that service instead of chasing database symptoms.",
        },
    ),
    "cache_poisoning_campaign": ScenarioBlueprint(
        scenario_id="cache_poisoning_campaign",
        name="Cache Poisoning Campaign",
        domain="content delivery",
        description="Cache poisoning via header confusion makes api-gateway look unstable while the cache is the real origin.",
        difficulty="medium",
        step_budget=8,
        root_cause_service="cache",
        root_cause_vector="Cache poisoning through signed-header confusion",
        causal_chain=("cache", "api-gateway", "database"),
        common_trap="Api-gateway errors spike first, but poisoned cache entries are driving the outage.",
        decoy_agent="log",
        decoy_service="api-gateway",
        best_followup_agent="infra",
        required_plan=(
            ("request_followup", None),
            ("isolate_service", "cache"),
            ("rollback_config", "cache"),
            ("restart_service", "cache"),
            ("submit_resolution", None),
        ),
        specialist_signal_map={
            "infra": "Cache saturation and latency reveal the real origin if inspected closely.",
            "log": "Logs emphasize gateway failures and can mislead toward api-gateway.",
            "security": "Security sees header replay activity but not the full runtime blast radius.",
        },
        cpu_pct={"api-gateway": 83.0, "database": 51.0, "cache": 91.0, "worker": 24.0, "auth_service": 20.0},
        mem_pct={"api-gateway": 74.0, "database": 44.0, "cache": 88.0, "worker": 21.0, "auth_service": 19.0},
        latency_ms={"api-gateway": 700.0, "database": 240.0, "cache": 860.0, "worker": 80.0, "auth_service": 95.0},
        error_rate={"api-gateway": 0.29, "database": 0.12, "cache": 0.39, "worker": 0.01, "auth_service": 0.01},
        health={"api-gateway": "critical", "database": "degraded", "cache": "critical", "worker": "healthy", "auth_service": "healthy"},
        log_lines=[
            "ERROR: gateway served poisoned cache variant to high-priority clients",
            "WARN: response signature mismatch on cached objects",
            "ERROR: cache deserialization fallback path triggered repeatedly",
        ],
        error_types={"GatewayVariantMismatch": 520, "SignatureMismatch": 218, "CacheDeserializeFallback": 611},
        stack_traces=["KeyError: poisoned_variant_signature missing in cache blob"],
        query_patterns=["GET /content?variant=admin", "X-Signed-Preview: replayed"],
        event_sequence=[
            _event("cache", "Cache accepted poisoned variants with replayed signed headers", "critical"),
            _event("api-gateway", "Gateway is serving poisoned variants and failing request validation", "critical"),
            _event("database", "Database load climbing after cache misses", "warning"),
        ],
        alert_strings=[
            "SECURITY_ALERT: HEADER_REPLAY detected on cache edge",
            "SECURITY_ALERT: CACHE_POISONING pattern detected",
        ],
        auth_fail_count=0,
        suspicious_ips=["198.51.100.42", "198.51.100.43"],
        injection_patterns=["CACHE_POISON: header_confusion", "HEADER_REPLAY: signed-preview"],
        cve_flags=["CWE-349: Acceptance of extraneous signed headers"],
        followup_notes={
            "infra": "Cache is hotter than gateway and is generating the downstream misses.",
            "log": "The logs make gateway look guilty, but every failing response references a poisoned cache variant.",
            "security": "Header replay exists, but infra is better at proving cache is the service to contain first.",
        },
    ),
}


def list_scenarios() -> list[ScenarioBlueprint]:
    return list(SCENARIOS.values())


def get_scenario(scenario_id: str) -> ScenarioBlueprint:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario_id: {scenario_id}")
    return SCENARIOS[scenario_id]


def scenarios_for_difficulty(difficulty: Difficulty) -> list[ScenarioBlueprint]:
    matched = [blueprint for blueprint in SCENARIOS.values() if blueprint.difficulty == difficulty]
    if not matched:
        raise ValueError(f"No scenario for difficulty {difficulty}")
    return matched


def scenario_for_difficulty(difficulty: Difficulty, seed: int | None = None) -> ScenarioBlueprint:
    matched = scenarios_for_difficulty(difficulty)
    if seed is None:
        return matched[0]
    return matched[seed % len(matched)]


def baseline_plan_for_scenario(scenario_id: str) -> list[dict[str, Any]]:
    blueprint = get_scenario(scenario_id)
    steps: list[dict[str, Any]] = []
    for action_type, target in blueprint.required_plan:
        if action_type == "request_followup":
            steps.append(
                {
                    "action_type": action_type,
                    "target_agent": blueprint.best_followup_agent,
                    "reasoning": "Ask the strongest specialist for clarification before acting.",
                }
            )
        elif action_type == "submit_resolution":
            steps.append(
                {
                    "action_type": action_type,
                    "resolution_summary": (
                        f"Contained and remediated {blueprint.root_cause_service} after confirming {blueprint.root_cause_vector}."
                    ),
                }
            )
        else:
            steps.append(
                {
                    "action_type": action_type,
                    "target_service": target,
                    "reasoning": f"Execute the planned {action_type} step on {target}.",
                }
            )
    return steps


def _jitter_metrics(values: dict[ServiceName, float], rng: random.Random) -> dict[ServiceName, float]:
    jittered: dict[ServiceName, float] = {}
    for service, value in values.items():
        delta = 0.0 if value == 0 else rng.uniform(-0.03, 0.03) * value
        jittered[service] = round(max(0.0, value + delta), 2)
    return jittered


def build_master_state(
    *,
    difficulty: Difficulty | None = None,
    scenario_id: str | None = None,
    seed: int | None = None,
    episode_id: str | None = None,
) -> MasterSREState:
    if scenario_id is not None:
        blueprint = get_scenario(scenario_id)
    elif difficulty is not None:
        blueprint = scenario_for_difficulty(difficulty, seed=seed)
    else:
        blueprint = scenario_for_difficulty("easy", seed=seed)

    rng = random.Random(seed if seed is not None else blueprint.step_budget)
    objective = ScenarioObjective(
        scenario_id=blueprint.scenario_id,
        name=blueprint.name,
        domain=blueprint.domain,
        description=blueprint.description,
        difficulty=blueprint.difficulty,
        root_cause_service=blueprint.root_cause_service,
        root_cause_vector=blueprint.root_cause_vector,
        decoy_agent=blueprint.decoy_agent,
        decoy_service=blueprint.decoy_service,
        best_followup_agent=blueprint.best_followup_agent,
        required_plan=list(blueprint.required_plan),
        common_trap=blueprint.common_trap,
        specialist_signal_map=copy.deepcopy(blueprint.specialist_signal_map),
    )
    state = MasterSREState(
        objective=objective,
        episode_id=episode_id or str(uuid.uuid4()),
        step_count=0,
        tick=0,
        step_budget=blueprint.step_budget,
        total_step_budget=blueprint.step_budget,
        workflow_stage="triage",
        cpu_pct=_jitter_metrics(copy.deepcopy(blueprint.cpu_pct), rng),
        mem_pct=_jitter_metrics(copy.deepcopy(blueprint.mem_pct), rng),
        latency_ms=_jitter_metrics(copy.deepcopy(blueprint.latency_ms), rng),
        error_rate={
            service: round(max(0.0, min(1.0, value + rng.uniform(-0.01, 0.01))), 3)
            for service, value in blueprint.error_rate.items()
        },
        health=copy.deepcopy(blueprint.health),
        log_lines=copy.deepcopy(blueprint.log_lines),
        error_types=copy.deepcopy(blueprint.error_types),
        stack_traces=copy.deepcopy(blueprint.stack_traces),
        query_patterns=copy.deepcopy(blueprint.query_patterns),
        event_sequence=copy.deepcopy(blueprint.event_sequence),
        alert_strings=copy.deepcopy(blueprint.alert_strings),
        auth_fail_count=blueprint.auth_fail_count,
        suspicious_ips=copy.deepcopy(blueprint.suspicious_ips),
        injection_patterns=copy.deepcopy(blueprint.injection_patterns),
        cve_flags=copy.deepcopy(blueprint.cve_flags),
        service_graph=copy.deepcopy(DEFAULT_SERVICE_GRAPH),
        causal_chain=list(blueprint.causal_chain),
        followup_answers_seen={agent: 0 for agent in AGENT_NAMES},
    )
    state.refresh_progress_flags()
    return state
