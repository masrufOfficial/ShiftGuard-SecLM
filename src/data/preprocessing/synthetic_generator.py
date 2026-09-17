"""Controlled Synthetic Security Scenario Generator for ShiftGuard-SecLM (V2).

Generates diverse, ground-truth-validated security reasoning instances across 16 application archetypes,
10 programming languages, and 3 prompt styles (ambiguous, functional-only, and security-aware),
supporting all 12 ShiftGuard multi-task reasoning capabilities.
"""

from __future__ import annotations
import random
import uuid
from typing import List, Dict, Any, Optional

from datasets.schemas.security_sample import (
    SecuritySample,
    TaskType,
    ConfidenceLevel,
    SeverityLevel,
    VerificationStatus,
    SecurityFinding,
    SecurityGroundTruth,
    RiskAssessment,
    AgentStep,
    ProvenanceMetadata,
)

ARCHETYPES = [
    "banking",
    "healthcare",
    "ecommerce",
    "auth_service",
    "education",
    "saas_admin",
    "payment_gateway",
    "file_upload",
    "admin_dashboard",
    "rest_api",
    "graphql_api",
    "microservice",
    "iot_backend",
    "cloud_webapp",
    "database_app",
    "chat_app",
]

ARCHETYPE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "banking": {
        "routes": ["/api/v1/transfer", "/finance/wire", "/accounts/send", "/api/v2/pay"],
        "src_vars": ["account_from", "source_acc", "sender_id", "from_account"],
        "dst_vars": ["account_to", "dest_acc", "recipient_id", "to_account"],
        "lang": "python",
        "framework": "fastapi",
        "database": "postgresql",
        "api_type": "REST",
        "threats": ["SQL Injection", "Broken Object Level Authorization (BOLA)", "Race Condition / Double Spending"],
        "cwe": ["CWE-89", "CWE-862", "CWE-362"],
        "owasp": ["A03:2021-Injection", "A01:2021-Broken Access Control"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.4,
        "requirements": [
            "Authenticate caller identity and verify authorization against source account.",
            "Use parameterized SQL queries or ORM models to prevent injection.",
            "Enforce ACID database transactions with row-level locks to prevent double spending.",
        ],
        "capabilities": ["ShiftGuard-StaticAnalyzer", "ShiftGuard-AccessAuditor", "ShiftGuard-RemediationEngine"],
        "agent_plan": [
            AgentStep(agent="auth_agent", action="verify_caller_ownership", priority=1),
            AgentStep(agent="injection_agent", action="audit_sql_parameterization", priority=2),
            AgentStep(agent="concurrency_agent", action="verify_database_transaction_locking", priority=3),
        ],
    },
    "healthcare": {
        "routes": ["/patients/records", "/medical/history", "/v1/diagnoses", "/health/ehr"],
        "src_vars": ["patient_id", "mrn", "health_id", "subject_id"],
        "dst_vars": ["notes", "diagnosis", "prescription", "lab_result"],
        "lang": "python",
        "framework": "django",
        "database": "postgresql",
        "api_type": "REST",
        "threats": ["Sensitive Data Exposure (PHI)", "Insecure Direct Object Reference (IDOR)", "Unencrypted Data at Rest"],
        "cwe": ["CWE-200", "CWE-862", "CWE-311"],
        "owasp": ["A01:2021-Broken Access Control", "A02:2021-Cryptographic Failures"],
        "severity": SeverityLevel.HIGH,
        "score": 8.8,
        "requirements": [
            "Encrypt Protected Health Information (PHI) at rest using AES-256-GCM.",
            "Enforce strict role-based access control (RBAC) permitting only assigned clinicians.",
            "Maintain tamper-evident audit logs for all medical record access events.",
        ],
        "capabilities": ["ShiftGuard-PrivacyAuditor", "ShiftGuard-AccessAuditor"],
        "agent_plan": [
            AgentStep(agent="privacy_agent", action="audit_phi_encryption_at_rest", priority=1),
            AgentStep(agent="access_agent", action="verify_clinician_patient_assignment", priority=2),
        ],
    },
    "ecommerce": {
        "routes": ["/cart/checkout", "/order/place", "/api/v1/purchase", "/shop/pay"],
        "src_vars": ["cart_id", "order_id", "basket_uuid", "session_cart"],
        "dst_vars": ["unit_price", "discount_code", "total_amt", "currency"],
        "lang": "javascript",
        "framework": "express",
        "database": "mongodb",
        "api_type": "REST",
        "threats": ["Client-Side Price Parameter Tampering", "Coupon Race Condition", "Broken Object Property Level Authorization"],
        "cwe": ["CWE-20", "CWE-362", "CWE-862"],
        "owasp": ["A04:2021-Insecure Design", "A01:2021-Broken Access Control"],
        "severity": SeverityLevel.HIGH,
        "score": 8.2,
        "requirements": [
            "Recalculate order prices strictly server-side from catalog database; ignore client price submissions.",
            "Use transactional atomic locks during discount code consumption.",
            "Enforce state transition integrity in checkout workflows.",
        ],
        "capabilities": ["ShiftGuard-BusinessLogicAuditor", "ShiftGuard-DataValidator"],
        "agent_plan": [
            AgentStep(agent="business_logic_agent", action="audit_server_side_price_recomputation", priority=1),
            AgentStep(agent="race_agent", action="verify_atomic_coupon_redemption", priority=2),
        ],
    },
    "auth_service": {
        "routes": ["/login", "/auth/token", "/v1/authenticate", "/user/signin"],
        "src_vars": ["username", "user_email", "login_id", "principal"],
        "dst_vars": ["password", "secret", "user_pass", "token"],
        "lang": "python",
        "framework": "fastapi",
        "database": "redis",
        "api_type": "REST",
        "threats": ["Weak Cryptographic Hash (MD5)", "Credential Stuffing", "Brute-Force Attacks"],
        "cwe": ["CWE-327", "CWE-287", "CWE-798"],
        "owasp": ["A02:2021-Cryptographic Failures", "A07:2021-Identification and Authentication Failures"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.1,
        "requirements": [
            "Use modern slow hashing algorithm (Argon2id or bcrypt with work factor >= 12).",
            "Implement rate limiting per IP and per username.",
            "Use constant-time comparison to prevent timing side-channel attacks.",
        ],
        "capabilities": ["ShiftGuard-CryptoAuditor", "ShiftGuard-RateLimitValidator"],
        "agent_plan": [
            AgentStep(agent="crypto_agent", action="verify_password_hashing_algorithm", priority=1),
            AgentStep(agent="rate_limit_agent", action="audit_rate_limiting_middleware", priority=2),
        ],
    },
    "education": {
        "routes": ["/grades/submit", "/exam/score", "/student/transcript", "/quiz/evaluate"],
        "src_vars": ["student_id", "exam_id", "assignment_id", "enrollment_id"],
        "dst_vars": ["grade_value", "feedback", "gpa", "override_flag"],
        "lang": "java",
        "framework": "spring",
        "database": "mysql",
        "api_type": "REST",
        "threats": ["Privilege Escalation (Grade Modification)", "Missing Function Level Authorization", "Insecure Session Binding"],
        "cwe": ["CWE-862", "CWE-285", "CWE-200"],
        "owasp": ["A01:2021-Broken Access Control", "A07:2021-Identification and Authentication Failures"],
        "severity": SeverityLevel.HIGH,
        "score": 8.1,
        "requirements": [
            "Assert instructor role binding to specific course section before allowing grade submission.",
            "Log all grade modification events with cryptographic checksums.",
        ],
        "capabilities": ["ShiftGuard-AccessAuditor", "ShiftGuard-AuditLogger"],
        "agent_plan": [
            AgentStep(agent="rbac_agent", action="verify_instructor_course_enrollment", priority=1),
        ],
    },
    "saas_admin": {
        "routes": ["/admin/tenant/billing", "/org/switch", "/api/tenants/config", "/saas/portal"],
        "src_vars": ["tenant_id", "organization_uuid", "account_key", "workspace_id"],
        "dst_vars": ["subscription_tier", "max_seats", "api_quota", "enterprise_features"],
        "lang": "python",
        "framework": "flask",
        "database": "postgresql",
        "api_type": "REST",
        "threats": ["Multi-Tenant Data Isolation Breach", "Tenant ID Spoofing", "Mass Assignment Vulnerability"],
        "cwe": ["CWE-862", "CWE-915", "CWE-200"],
        "owasp": ["A01:2021-Broken Access Control", "A04:2021-Insecure Design"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.3,
        "requirements": [
            "Derive tenant_id strictly from verified JWT claim; reject tenant_id from client request body or query parameter.",
            "Enforce PostgreSQL Row Level Security (RLS) across all tenant queries.",
        ],
        "capabilities": ["ShiftGuard-MultiTenantAuditor", "ShiftGuard-AccessAuditor"],
        "agent_plan": [
            AgentStep(agent="tenant_agent", action="verify_jwt_tenant_derivation", priority=1),
            AgentStep(agent="db_agent", action="audit_row_level_security_policies", priority=2),
        ],
    },
    "payment_gateway": {
        "routes": ["/v1/charge", "/payments/capture", "/checkout/process", "/card/authorize"],
        "src_vars": ["card_token", "merchant_id", "transaction_ref", "idempotency_key"],
        "dst_vars": ["amount_cents", "cvv_hash", "currency", "payment_method"],
        "lang": "go",
        "framework": "gin",
        "database": "redis",
        "api_type": "REST",
        "threats": ["Replay Attacks", "Missing Idempotency Control", "PCI-DSS Plaintext Card Data Logging"],
        "cwe": ["CWE-294", "CWE-532", "CWE-362"],
        "owasp": ["A04:2021-Insecure Design", "A09:2021-Security Logging and Monitoring Failures"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.5,
        "requirements": [
            "Enforce atomic idempotency key validation with Redis SETNX before calling payment processor.",
            "Mask primary account numbers (PAN) and scrub CVV values from all application logs.",
        ],
        "capabilities": ["ShiftGuard-ReplayAuditor", "ShiftGuard-SecretScrubber"],
        "agent_plan": [
            AgentStep(agent="idempotency_agent", action="verify_redis_atomic_lock", priority=1),
            AgentStep(agent="pci_agent", action="audit_card_data_masking", priority=2),
        ],
    },
    "file_upload": {
        "routes": ["/upload", "/profile/avatar", "/documents/attach", "/media/store"],
        "src_vars": ["filename", "file_name", "target_doc", "asset_name"],
        "dst_vars": ["upload_dir", "dest_path", "storage_folder", "asset_dir"],
        "lang": "javascript",
        "framework": "express",
        "database": "filesystem",
        "api_type": "REST",
        "threats": ["Path Traversal", "Unrestricted File Upload", "Arbitrary Code Execution"],
        "cwe": ["CWE-22", "CWE-434"],
        "owasp": ["A01:2021-Broken Access Control", "A04:2021-Insecure Design"],
        "severity": SeverityLevel.HIGH,
        "score": 8.6,
        "requirements": [
            "Sanitize filename using server-generated UUIDs; do not trust user-supplied filenames.",
            "Verify MIME types and magic bytes against a strict whitelist.",
            "Store uploaded content outside the web root or disable script execution.",
        ],
        "capabilities": ["ShiftGuard-FileAuditor", "ShiftGuard-PathTraversalChecker"],
        "agent_plan": [
            AgentStep(agent="path_agent", action="verify_directory_containment", priority=1),
            AgentStep(agent="mime_agent", action="inspect_file_type_validation", priority=2),
        ],
    },
    "admin_dashboard": {
        "routes": ["/admin/users/role", "/control/elevate", "/api/admin/permissions", "/system/roles"],
        "src_vars": ["target_user", "target_uid", "member_id", "account_id"],
        "dst_vars": ["new_role", "is_admin", "permissions_mask", "role_level"],
        "lang": "csharp",
        "framework": "aspnet",
        "database": "sqlserver",
        "api_type": "REST",
        "threats": ["Vertical Privilege Escalation", "Missing Function-Level Access Control", "CSRF State Mutation"],
        "cwe": ["CWE-862", "CWE-285", "CWE-352"],
        "owasp": ["A01:2021-Broken Access Control"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.2,
        "requirements": [
            "Validate administrator role via secure server-side session or signed claims.",
            "Enforce anti-CSRF token verification on all POST/PUT privilege elevation endpoints.",
        ],
        "capabilities": ["ShiftGuard-AccessAuditor", "ShiftGuard-CSRFValidator"],
        "agent_plan": [
            AgentStep(agent="auth_agent", action="verify_admin_role_authorization", priority=1),
            AgentStep(agent="csrf_agent", action="audit_anti_forgery_token_presence", priority=2),
        ],
    },
    "rest_api": {
        "routes": ["/api/v1/items", "/v2/resources", "/api/query", "/service/fetch"],
        "src_vars": ["filter_param", "sort_by", "search_term", "query_str"],
        "dst_vars": ["page_size", "limit", "offset", "fields"],
        "lang": "python",
        "framework": "fastapi",
        "database": "sqlite",
        "api_type": "REST",
        "threats": ["SQL Injection via Dynamic Order By", "Resource Exhaustion via Unbounded Pagination"],
        "cwe": ["CWE-89", "CWE-770"],
        "owasp": ["A03:2021-Injection", "A04:2021-Insecure Design"],
        "severity": SeverityLevel.HIGH,
        "score": 8.4,
        "requirements": [
            "Whitelist sort columns against allowed schema attributes.",
            "Cap maximum pagination limit to 100 records.",
        ],
        "capabilities": ["ShiftGuard-StaticAnalyzer", "ShiftGuard-RateLimitValidator"],
        "agent_plan": [
            AgentStep(agent="injection_agent", action="audit_order_by_whitelisting", priority=1),
        ],
    },
    "graphql_api": {
        "routes": ["/graphql", "/api/graphql", "/v1/gql", "/query"],
        "src_vars": ["query", "variables", "operation_name", "document"],
        "dst_vars": ["depth_limit", "cost_analysis", "complexity_score", "field_alias"],
        "lang": "typescript",
        "framework": "apollo",
        "database": "postgresql",
        "api_type": "GraphQL",
        "threats": ["GraphQL Circular Query DoS", "GraphQL Field-Level Authorization Bypass", "Batching Attacks"],
        "cwe": ["CWE-770", "CWE-862"],
        "owasp": ["A04:2021-Insecure Design", "A01:2021-Broken Access Control"],
        "severity": SeverityLevel.HIGH,
        "score": 8.5,
        "requirements": [
            "Enforce query depth limiting (max depth 6) and query complexity analysis.",
            "Apply field-level resolver authorization checks.",
        ],
        "capabilities": ["ShiftGuard-GraphQLAuditor", "ShiftGuard-AccessAuditor"],
        "agent_plan": [
            AgentStep(agent="graphql_agent", action="verify_query_depth_and_complexity", priority=1),
        ],
    },
    "microservice": {
        "routes": ["/internal/sync", "/rpc/notify", "/event/dispatch", "/worker/consume"],
        "src_vars": ["event_payload", "routing_key", "message_body", "rpc_request"],
        "dst_vars": ["destination_queue", "target_service", "ack_mode", "retry_count"],
        "lang": "go",
        "framework": "grpc",
        "database": "rabbitmq",
        "api_type": "gRPC",
        "threats": ["Server-Side Request Forgery (SSRF)", "Unauthenticated Internal RPC", "Insecure Deserialization"],
        "cwe": ["CWE-918", "CWE-306", "CWE-502"],
        "owasp": ["A10:2021-Server-Side Request Forgery", "A07:2021-Identification and Authentication Failures"],
        "severity": SeverityLevel.HIGH,
        "score": 8.7,
        "requirements": [
            "Require mutual TLS (mTLS) for all inter-service gRPC RPC communications.",
            "Validate destination service hostnames against internal service discovery registry.",
        ],
        "capabilities": ["ShiftGuard-NetworkAuditor", "ShiftGuard-TransportSecurity"],
        "agent_plan": [
            AgentStep(agent="mtls_agent", action="verify_grpc_tls_credentials", priority=1),
        ],
    },
    "iot_backend": {
        "routes": ["/device/telemetry", "/firmware/update", "/sensor/metric", "/gateway/beacon"],
        "src_vars": ["device_id", "serial_number", "chip_uid", "firmware_hash"],
        "dst_vars": ["telemetry_raw", "command_payload", "sample_rate", "signature"],
        "lang": "c",
        "framework": "posix",
        "database": "influxdb",
        "api_type": "MQTT",
        "threats": ["Buffer Overflow in Protocol Parser", "Missing Firmware Signature Verification", "Cleartext Telemetry Transmission"],
        "cwe": ["CWE-120", "CWE-347", "CWE-319"],
        "owasp": ["A06:2021-Vulnerable and Outdated Components", "A02:2021-Cryptographic Failures"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.6,
        "requirements": [
            "Enforce strict bounds checking in binary packet decoders.",
            "Verify cryptographic signatures (RSA-PSS / Ed25519) before applying firmware updates.",
        ],
        "capabilities": ["ShiftGuard-MemoryAuditor", "ShiftGuard-CryptoAuditor"],
        "agent_plan": [
            AgentStep(agent="bounds_agent", action="verify_packet_bounds_checks", priority=1),
            AgentStep(agent="sig_agent", action="audit_firmware_signature_verification", priority=2),
        ],
    },
    "cloud_webapp": {
        "routes": ["/profile/bio", "/comments/new", "/forum/post", "/feedback/send"],
        "src_vars": ["user_input", "post_body", "author_comment", "html_content"],
        "dst_vars": ["sanitized_text", "rendered_dom", "markdown_preview", "safe_html"],
        "lang": "typescript",
        "framework": "nextjs",
        "database": "postgresql",
        "api_type": "REST",
        "threats": ["Stored Cross-Site Scripting (XSS)", "HTML Injection", "Missing Content Security Policy"],
        "cwe": ["CWE-79", "CWE-16"],
        "owasp": ["A03:2021-Injection", "A05:2021-Security Misconfiguration"],
        "severity": SeverityLevel.HIGH,
        "score": 8.0,
        "requirements": [
            "Sanitize rich HTML input using DOMPurify before rendering into dangerouslySetInnerHTML.",
            "Configure strict Content-Security-Policy (CSP) headers without 'unsafe-inline'.",
        ],
        "capabilities": ["ShiftGuard-XSSDetector", "ShiftGuard-HeaderAuditor"],
        "agent_plan": [
            AgentStep(agent="xss_agent", action="audit_html_sanitization", priority=1),
        ],
    },
    "database_app": {
        "routes": ["/reports/custom", "/analytics/aggregate", "/data/export", "/stats/generate"],
        "src_vars": ["table_name", "where_clause", "group_field", "raw_sql"],
        "dst_vars": ["result_set", "export_format", "row_count", "cached_query"],
        "lang": "sql",
        "framework": "psycopg2",
        "database": "postgresql",
        "api_type": "CLI",
        "threats": ["Second-Order SQL Injection", "Arbitrary Table Read", "Database Superuser Privilege Abuse"],
        "cwe": ["CWE-89", "CWE-250"],
        "owasp": ["A03:2021-Injection", "A01:2021-Broken Access Control"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.4,
        "requirements": [
            "Use sql.Identifier and sql.Placeholder constructs instead of string interpolation.",
            "Connect using dedicated read-only database service account with minimal table grants.",
        ],
        "capabilities": ["ShiftGuard-StaticAnalyzer", "ShiftGuard-PrivilegeAuditor"],
        "agent_plan": [
            AgentStep(agent="sql_agent", action="verify_query_parameterization", priority=1),
        ],
    },
    "chat_app": {
        "routes": ["/ws/chat", "/messages/send", "/channel/stream", "/room/broadcast"],
        "src_vars": ["channel_id", "msg_payload", "sender_jwt", "recipient_id"],
        "dst_vars": ["channel_members", "chat_history", "broadcast_queue", "socket_id"],
        "lang": "javascript",
        "framework": "websocket",
        "database": "redis",
        "api_type": "WebSocket",
        "threats": ["WebSocket Connection Hijacking", "Cross-Channel Message Injection", "Missing Channel Membership Authorization"],
        "cwe": ["CWE-862", "CWE-287"],
        "owasp": ["A01:2021-Broken Access Control", "A07:2021-Identification and Authentication Failures"],
        "severity": SeverityLevel.HIGH,
        "score": 8.3,
        "requirements": [
            "Validate authenticated user's active membership in channel_id before broadcasting messages.",
            "Verify origin header on WebSocket connection handshake to prevent CSWSH.",
        ],
        "capabilities": ["ShiftGuard-AccessAuditor", "ShiftGuard-HandshakeValidator"],
        "agent_plan": [
            AgentStep(agent="ws_agent", action="verify_websocket_origin_handshake", priority=1),
        ],
    },
}

PROMPT_STYLES = {
    "ambiguous": [
        "Write code to handle {route} quickly.",
        "Take {src_var} from the incoming request and query {dst_var}.",
        "Implement {route} without unnecessary boilerplate.",
        "Just process {src_var} and return the result as JSON.",
    ],
    "functional_only": [
        "Implement a {framework} route at {route} that accepts {src_var} and updates {dst_var}.",
        "Create the endpoint {route} in {lang} to process {src_var} and save to the database.",
        "Build a {framework} service handling {route} for {archetype} operations.",
    ],
    "security_aware": [
        "Implement a secure {framework} endpoint for {route} with proper authentication, parameter validation for {src_var}, and defense against {cwe}.",
        "Build a production-hardened {route} handler in {lang} enforcing least privilege, input sanitization, and compliance with OWASP {owasp}.",
        "Develop a robust {route} endpoint preventing unauthorized tampering with {dst_var}.",
    ],
}


class SyntheticSecurityGenerator:
    """Generates synthetic security reasoning samples across 16 application archetypes."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_sample(
        self,
        archetype: str,
        task: TaskType,
        scenario_idx: int = 0,
        prompt_style: str = "functional_only",
    ) -> SecuritySample:
        tpl = ARCHETYPE_TEMPLATES[archetype]
        route = self.rng.choice(tpl["routes"])
        src_var = self.rng.choice(tpl["src_vars"])
        dst_var = self.rng.choice(tpl["dst_vars"])
        lang = tpl["lang"]
        framework = tpl["framework"]
        database = tpl["database"]
        api_type = tpl["api_type"]
        cwe_primary = tpl["cwe"][0]
        owasp_primary = tpl["owasp"][0]

        # Format prompt according to requested style
        prompt_template = self.rng.choice(PROMPT_STYLES[prompt_style])
        prompt = prompt_template.format(
            route=route,
            src_var=src_var,
            dst_var=dst_var,
            framework=framework,
            lang=lang,
            archetype=archetype,
            cwe=cwe_primary,
            owasp=owasp_primary,
        )

        requirement = f"Develop secure implementation for {archetype} route {route} adhering to {owasp_primary}."

        uid = str(uuid.uuid4())[:8]
        scenario_family = f"synth_{archetype}_scen{scenario_idx:04d}"

        # Code snippets (Vulnerable & Secure)
        if lang == "python":
            vuln_code = (
                f"@app.post('{route}')\n"
                f"def handle_{archetype}_{uid[:4]}({src_var}: str, {dst_var}: str):\n"
                f"    # Flaw: Unparameterized execution directly interpolating input\n"
                f"    query = f\"SELECT * FROM {archetype} WHERE {src_var} = '{{{src_var}}}'\"\n"
                f"    return db.execute(query).fetchall()\n"
            )
            repair_code = (
                f"@app.post('{route}')\n"
                f"def handle_{archetype}_{uid[:4]}({src_var}: str, {dst_var}: str, user = Depends(get_current_user)):\n"
                f"    # Fix: Parameterized query + ownership check\n"
                f"    query = 'SELECT * FROM {archetype} WHERE {src_var} = :src AND user_id = :uid'\n"
                f"    return db.execute(text(query), {{'src': {src_var}, 'uid': user.id}}).fetchall()\n"
            )
        elif lang in ("javascript", "typescript"):
            vuln_code = (
                f"app.post('{route}', (req, res) => {{\n"
                f"    const {{ {src_var}, {dst_var} }} = req.body;\n"
                f"    // Flaw: Direct string concatenation and missing authentication\n"
                f"    const sql = `SELECT * FROM {archetype} WHERE {src_var} = '${{{src_var}}}'`;\n"
                f"    db.query(sql, (err, rows) => res.json(rows));\n"
                f"}});\n"
            )
            repair_code = (
                f"app.post('{route}', authenticateToken, (req, res) => {{\n"
                f"    const {{ {src_var}, {dst_var} }} = req.body;\n"
                f"    // Fix: Parameterized SQL array bindings\n"
                f"    const sql = 'SELECT * FROM {archetype} WHERE {src_var} = $1 AND user_id = $2';\n"
                f"    db.query(sql, [{src_var}, req.user.id], (err, rows) => res.json(rows));\n"
                f"}});\n"
            )
        else:
            vuln_code = (
                f"func Handle_{archetype}_{uid[:4]}(c *gin.Context) {{\n"
                f"    {src_var} := c.PostForm(\"{src_var}\")\n"
                f"    // Flaw: Unchecked execution of raw input\n"
                f"    out, _ := exec.Command(\"sh\", \"-c\", {src_var}).Output()\n"
                f"    c.JSON(200, gin.H{{\"output\": string(out)}})\n"
                f"}}\n"
            )
            repair_code = (
                f"func Handle_{archetype}_{uid[:4]}(c *gin.Context) {{\n"
                f"    {src_var} := c.PostForm(\"{src_var}\")\n"
                f"    if !isValidInput({src_var}) {{\n"
                f"        c.JSON(400, gin.H{{\"error\": \"Invalid input parameter\"}})\n"
                f"        return\n"
                f"    }}\n"
                f"    c.JSON(200, gin.H{{\"status\": \"success\"}})\n"
                f"}}\n"
            )

        findings = [
            SecurityFinding(
                source="semgrep",
                rule_id=f"{lang}.{framework}.security.{cwe_primary.lower()}",
                cwe=cwe_primary,
                severity=tpl["severity"],
                line=4,
                message=f"Exposure matching {cwe_primary} detected in {route}.",
            )
        ]

        ground_truth = SecurityGroundTruth(
            threats=tpl["threats"],
            cwe=tpl["cwe"],
            owasp=tpl["owasp"],
            risk=RiskAssessment(
                severity=tpl["severity"],
                score=tpl["score"],
                confidence=0.95,
                explanation=f"Exposures in {archetype} handling endpoint {route}.",
            ),
            security_requirements=tpl["requirements"],
            security_strategy=[
                f"Enforce Shift-Left proactive controls for {archetype}.",
                "Validate input boundaries at route registration.",
            ],
            capability_plan=tpl["capabilities"],
            recommended_capabilities=tpl["capabilities"],
            agent_plan=tpl["agent_plan"],
            explanation=f"Endpoint {route} improperly handles `{src_var}` without parameterization or access controls.",
            repair_guidance=[f"Remediate {c} by applying parameterization and access controls." for c in tpl["cwe"]],
            repair_code=repair_code,
            verification=VerificationStatus.FIXED,
        )

        sample = SecuritySample(
            sample_id=f"synth-{archetype}-{task.value}-{uid}",
            source="synthetic_shiftguard",
            source_id=scenario_family,
            application_archetype=archetype,
            domain="enterprise_software",
            repository=f"shiftguard/{archetype}",
            language=lang,
            framework=framework,
            database=database,
            api_type=api_type,
            requirement=requirement,
            developer_prompt=prompt,
            project_context={
                "language": lang,
                "framework": framework,
                "database": database,
                "api_type": api_type,
                "archetype": archetype,
                "route": route,
                "prompt_style": prompt_style,
            },
            source_code=vuln_code,
            security_findings=findings,
            threat_description=tpl["threats"][0],
            security_requirements=tpl["requirements"],
            cwe=tpl["cwe"],
            cwe_description=f"Standard flaw in {archetype} processing",
            owasp_category=tpl["owasp"],
            risk_level=tpl["severity"].value,
            severity=tpl["severity"].value,
            confidence=ConfidenceLevel.HIGH,
            security_strategy=[f"Enforce Shift-Left proactive controls for {archetype}."],
            capability_plan=tpl["capabilities"],
            agent_plan=tpl["agent_plan"],
            vulnerability_explanation=f"Flaw identified in {archetype} route {route}.",
            secure_repair=repair_code,
            verification=VerificationStatus.FIXED,
            task_type=task,
            provenance=ProvenanceMetadata(
                source_name="synthetic_shiftguard",
                source_id=scenario_family,
                repository=f"shiftguard/{archetype}",
                license="Apache-2.0",
                is_synthetic=True,
                evidence_source="controlled_synthetic_generator",
                archetype=archetype,
            ),
            license="Apache-2.0",
        )

        return sample

    def generate_corpus(self, num_scenarios_per_archetype: int = 15) -> List[SecuritySample]:
        """Generates a rich multi-task dataset across all 16 archetypes."""
        samples = []
        tasks = list(TaskType)
        prompt_styles = ["ambiguous", "functional_only", "security_aware"]

        for arch in ARCHETYPE_TEMPLATES.keys():
            for scen_idx in range(num_scenarios_per_archetype):
                style = prompt_styles[scen_idx % len(prompt_styles)]
                # For each scenario, generate records covering tasks
                for task in tasks:
                    sample = self.generate_sample(
                        archetype=arch,
                        task=task,
                        scenario_idx=scen_idx,
                        prompt_style=style,
                    )
                    samples.append(sample)
        return samples
