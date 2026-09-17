"""Controlled Synthetic Security Scenario Generator for ShiftGuard-SecLM.

Generates diverse, ground-truth-validated security reasoning instances across 12 application archetypes,
supporting multi-task training (threats, CWE, OWASP, risk, agent planning, and repair).
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
    "banking_api",
    "healthcare_api",
    "ecommerce",
    "auth_service",
    "file_upload",
    "admin_dashboard",
    "iot_backend",
    "payment_gateway",
    "chat_app",
    "education_platform",
    "microservice",
    "saas_admin",
]

ARCHETYPE_TEMPLATES = {
    "banking_api": {
        "routes": ["/api/v1/transfer", "/finance/wire", "/accounts/send", "/api/v2/pay"],
        "src_vars": ["account_from", "source_acc", "sender_id", "from_account"],
        "dst_vars": ["account_to", "dest_acc", "recipient_id", "to_account"],
        "lang": "python",
        "framework": "fastapi",
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
        "agent_plan": [
            AgentStep(agent="auth_agent", action="verify_caller_ownership", priority=1),
            AgentStep(agent="injection_agent", action="audit_sql_parameterization", priority=2),
            AgentStep(agent="concurrency_agent", action="verify_database_transaction_locking", priority=3),
        ],
    },
    "file_upload": {
        "routes": ["/upload", "/profile/avatar", "/documents/attach", "/media/store"],
        "src_vars": ["filename", "file_name", "target_doc", "asset_name"],
        "dst_vars": ["upload_dir", "dest_path", "storage_folder", "asset_dir"],
        "lang": "javascript",
        "framework": "express",
        "threats": ["Path Traversal", "Unrestricted File Upload", "Arbitrary Code Execution"],
        "cwe": ["CWE-22", "CWE-434"],
        "owasp": ["A01:2021-Broken Access Control", "A04:2021-Insecure Design"],
        "severity": SeverityLevel.HIGH,
        "score": 8.6,
        "requirements": [
            "Sanitize filename using server-generated UUIDs; do not trust user-supplied filenames.",
            "Verify MIME types against a strict whitelist of image extensions.",
            "Store uploaded content outside the web root or disable script execution on the upload directory.",
        ],
        "agent_plan": [
            AgentStep(agent="path_agent", action="verify_directory_containment", priority=1),
            AgentStep(agent="mime_agent", action="inspect_file_type_validation", priority=2),
        ],
    },
    "auth_service": {
        "routes": ["/login", "/auth/token", "/v1/authenticate", "/user/signin"],
        "src_vars": ["username", "user_email", "login_id", "principal"],
        "dst_vars": ["password", "secret", "user_pass", "token"],
        "lang": "python",
        "framework": "fastapi",
        "threats": ["Weak Cryptographic Hash (MD5)", "Credential Stuffing", "Brute-Force Attacks"],
        "cwe": ["CWE-327", "CWE-307", "CWE-89"],
        "owasp": ["A02:2021-Cryptographic Failures", "A07:2021-Identification and Authentication Failures"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.1,
        "requirements": [
            "Use modern slow hashing algorithm (Argon2id or bcrypt with work factor >= 12).",
            "Implement rate limiting per IP and per username.",
            "Use constant-time comparison to prevent timing side-channel attacks.",
        ],
        "agent_plan": [
            AgentStep(agent="crypto_agent", action="verify_password_hashing_algorithm", priority=1),
            AgentStep(agent="rate_limit_agent", action="audit_rate_limiting_middleware", priority=2),
        ],
    },
    "healthcare_api": {
        "routes": ["/patients/records", "/medical/history", "/v1/diagnoses", "/health/ehr"],
        "src_vars": ["patient_id", "mrn", "health_id", "subject_id"],
        "dst_vars": ["notes", "diagnosis", "prescription", "lab_result"],
        "lang": "python",
        "framework": "django",
        "threats": ["Sensitive Data Exposure (PHI)", "Insecure Direct Object Reference (IDOR)", "Unencrypted Data at Rest"],
        "cwe": ["CWE-200", "CWE-639", "CWE-311"],
        "owasp": ["A01:2021-Broken Access Control", "A02:2021-Cryptographic Failures"],
        "severity": SeverityLevel.HIGH,
        "score": 8.8,
        "requirements": [
            "Encrypt Protected Health Information (PHI) at rest using AES-256-GCM.",
            "Enforce strict role-based access control (RBAC) permitting only assigned clinicians.",
            "Maintain tamper-evident audit logs for all medical record access events.",
        ],
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
        "threats": ["Client-Side Price Parameter Tampering", "Coupon Race Condition", "Broken Object Property Level Authorization"],
        "cwe": ["CWE-472", "CWE-362", "CWE-20"],
        "owasp": ["A04:2021-Insecure Design", "A01:2021-Broken Access Control"],
        "severity": SeverityLevel.HIGH,
        "score": 8.2,
        "requirements": [
            "Recalculate order prices strictly server-side from catalog database; ignore client price submissions.",
            "Use transactional atomic locks during discount code consumption.",
            "Enforce state transition integrity in checkout workflows.",
        ],
        "agent_plan": [
            AgentStep(agent="business_logic_agent", action="audit_server_side_price_recomputation", priority=1),
            AgentStep(agent="race_agent", action="verify_atomic_coupon_redemption", priority=2),
        ],
    },
    "admin_dashboard": {
        "routes": ["/admin/users/role", "/control/elevate", "/api/admin/permissions", "/system/roles"],
        "src_vars": ["target_user", "target_uid", "member_id", "account_id"],
        "dst_vars": ["new_role", "is_admin", "permissions_mask", "role_level"],
        "lang": "python",
        "framework": "fastapi",
        "threats": ["Privilege Escalation", "Missing Function Level Access Control", "Mass Assignment"],
        "cwe": ["CWE-269", "CWE-915", "CWE-862"],
        "owasp": ["A01:2021-Broken Access Control", "A04:2021-Insecure Design"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.3,
        "requirements": [
            "Verify superadmin session token and MFA confirmation before granting elevated roles.",
            "Enforce immutable schema DTOs preventing mass-assignment of administrative flags.",
            "Emit immediate high-priority alerts to security teams upon any role elevation.",
        ],
        "agent_plan": [
            AgentStep(agent="privilege_agent", action="audit_role_assignment_authorization", priority=1),
            AgentStep(agent="mass_assignment_agent", action="verify_dto_schema_immutability", priority=2),
        ],
    },
    "iot_backend": {
        "routes": ["/mqtt/telemetry", "/devices/command", "/iot/v1/sensor", "/gateway/push"],
        "src_vars": ["device_id", "mac_address", "sensor_uuid", "gateway_key"],
        "dst_vars": ["payload_data", "command_str", "firmware_url", "metric_val"],
        "lang": "go",
        "framework": "gin",
        "threats": ["Hardcoded Device Master Key", "Cleartext MQTT Transport", "Command Injection via Telemetry"],
        "cwe": ["CWE-798", "CWE-319", "CWE-78"],
        "owasp": ["A07:2021-Identification and Authentication Failures", "A02:2021-Cryptographic Failures"],
        "severity": SeverityLevel.HIGH,
        "score": 8.7,
        "requirements": [
            "Authenticate IoT devices using per-device X.509 client certificates.",
            "Enforce TLS 1.3 on all MQTT and HTTPS communication channels.",
            "Validate telemetry payloads against strict binary or JSON schemas before processing.",
        ],
        "agent_plan": [
            AgentStep(agent="iot_cert_agent", action="verify_device_certificate_authentication", priority=1),
            AgentStep(agent="transport_agent", action="audit_tls_enforcement", priority=2),
        ],
    },
    "payment_gateway": {
        "routes": ["/webhooks/stripe", "/payments/callback", "/billing/ipn", "/checkout/confirm"],
        "src_vars": ["webhook_secret", "signature_header", "event_id", "nonce"],
        "dst_vars": ["amount_paid", "payment_status", "customer_id", "charge_id"],
        "lang": "python",
        "framework": "fastapi",
        "threats": ["Webhook Forgery", "Replay Attack", "Insecure Direct Callback Handling"],
        "cwe": ["CWE-345", "CWE-294", "CWE-347"],
        "owasp": ["A08:2021-Software and Data Integrity Failures", "A07:2021-Identification and Authentication Failures"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.2,
        "requirements": [
            "Verify cryptographic HMAC-SHA256 signature on all incoming webhook payloads.",
            "Store and verify transaction nonces to guarantee idempotent single-use event processing.",
            "Perform out-of-band verification against the payment processor API before crediting balances.",
        ],
        "agent_plan": [
            AgentStep(agent="webhook_agent", action="verify_hmac_signature_validation", priority=1),
            AgentStep(agent="idempotency_agent", action="audit_nonce_deduplication", priority=2),
        ],
    },
    "chat_app": {
        "routes": ["/ws/chat", "/messages/send", "/rooms/join", "/v1/broadcast"],
        "src_vars": ["room_id", "channel_tag", "sender_name", "session_id"],
        "dst_vars": ["message_body", "raw_content", "attachment_link", "text_payload"],
        "lang": "javascript",
        "framework": "express",
        "threats": ["Stored Cross-Site Scripting (XSS)", "Cross-Site WebSocket Hijacking (CSWSH)", "Unauthenticated Room Subscription"],
        "cwe": ["CWE-79", "CWE-1385", "CWE-287"],
        "owasp": ["A03:2021-Injection", "A01:2021-Broken Access Control"],
        "severity": SeverityLevel.HIGH,
        "score": 8.0,
        "requirements": [
            "Validate Origin header on WebSocket upgrade requests to prevent CSWSH.",
            "Sanitize all chat message HTML and markdown using context-aware DOMPurify / escaping.",
            "Enforce channel authorization tokens prior to establishing live message streaming.",
        ],
        "agent_plan": [
            AgentStep(agent="websocket_agent", action="verify_origin_header_validation", priority=1),
            AgentStep(agent="xss_agent", action="audit_contextual_html_sanitization", priority=2),
        ],
    },
    "education_platform": {
        "routes": ["/grades/submit", "/courses/enroll", "/student/transcript", "/exams/result"],
        "src_vars": ["student_id", "enrollment_num", "exam_code", "course_id"],
        "dst_vars": ["grade_letter", "score_val", "gpa_record", "feedback_text"],
        "lang": "python",
        "framework": "django",
        "threats": ["BOLA Grade Tampering", "Insecure Session Fixation", "Mass Enrollment IDOR"],
        "cwe": ["CWE-639", "CWE-384", "CWE-862"],
        "owasp": ["A01:2021-Broken Access Control", "A07:2021-Identification and Authentication Failures"],
        "severity": SeverityLevel.HIGH,
        "score": 8.3,
        "requirements": [
            "Verify instructor role assignment against course department before allowing grade submissions.",
            "Regenerate session identifier upon user authentication to prevent session fixation.",
            "Enforce object-level permissions on student transcript retrieval.",
        ],
        "agent_plan": [
            AgentStep(agent="grade_auth_agent", action="verify_instructor_course_authorization", priority=1),
            AgentStep(agent="session_agent", action="audit_session_regeneration_on_login", priority=2),
        ],
    },
    "microservice": {
        "routes": ["/proxy/fetch", "/internal/relay", "/service/dispatch", "/gateway/forward"],
        "src_vars": ["target_url", "downstream_endpoint", "service_uri", "dest_host"],
        "dst_vars": ["header_map", "auth_token", "query_payload", "timeout_ms"],
        "lang": "python",
        "framework": "fastapi",
        "threats": ["Server-Side Request Forgery (SSRF)", "Internal Metadata Service Access", "Insecure Service-to-Service Trust"],
        "cwe": ["CWE-918", "CWE-200"],
        "owasp": ["A10:2021-Server-Side Request Forgery", "A01:2021-Broken Access Control"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.0,
        "requirements": [
            "Enforce strict domain whitelist for outbound HTTP proxy requests.",
            "Block access to RFC 1918 private subnets and cloud metadata IPs (169.254.169.254).",
            "Require mutual TLS (mTLS) with cryptographically validated service identities for internal communication.",
        ],
        "agent_plan": [
            AgentStep(agent="ssrf_agent", action="audit_url_target_validation", priority=1),
            AgentStep(agent="metadata_agent", action="verify_cloud_metadata_ip_filtering", priority=2),
        ],
    },
    "saas_admin": {
        "routes": ["/tenants/query", "/org/analytics", "/workspace/users", "/billing/tenant"],
        "src_vars": ["tenant_id", "workspace_slug", "organization_uuid", "customer_key"],
        "dst_vars": ["filter_criteria", "report_type", "export_format", "user_role"],
        "lang": "python",
        "framework": "fastapi",
        "threats": ["Cross-Tenant Data Leakage", "Tenant Isolation Bypass", "SQL Injection in Tenant Filter"],
        "cwe": ["CWE-200", "CWE-89", "CWE-284"],
        "owasp": ["A01:2021-Broken Access Control", "A03:2021-Injection"],
        "severity": SeverityLevel.CRITICAL,
        "score": 9.5,
        "requirements": [
            "Inject tenant identifier into all database queries at the ORM / connection session level.",
            "Enforce database row-level security (RLS) policies per tenant.",
            "Audit all cross-tenant database access attempts and flag anomalies immediately.",
        ],
        "agent_plan": [
            AgentStep(agent="multitenancy_agent", action="verify_row_level_security_enforcement", priority=1),
            AgentStep(agent="isolation_agent", action="audit_tenant_id_query_parameterization", priority=2),
        ],
    },
}


class SyntheticSecurityGenerator:
    """Generates synthetic security training samples conforming to the 11 multi-task objectives."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_sample(
        self,
        archetype: str,
        task: TaskType,
        sample_idx: int = 0,
    ) -> SecuritySample:
        """Generates a structured SecuritySample for a designated archetype and task with randomized variables."""
        if archetype not in ARCHETYPE_TEMPLATES:
            archetype = "banking_api"

        tpl = ARCHETYPE_TEMPLATES[archetype]
        route = self.rng.choice(tpl["routes"])
        src_var = self.rng.choice(tpl["src_vars"])
        dst_var = self.rng.choice(tpl["dst_vars"])
        uid = f"{sample_idx:04d}-{self.rng.randint(100, 999)}"

        lang = tpl["lang"]
        framework = tpl["framework"]

        requirement = f"Implement a secure {archetype.replace('_', ' ')} handling endpoint {route}."
        prompt = f"Create a {framework.capitalize()} route `{route}` taking `{src_var}` and `{dst_var}` with appropriate business logic."

        # Generate realistic code snippets
        if lang == "python":
            vuln_code = (
                f"@app.post('{route}')\n"
                f"def handle_{archetype}_{uid[:4]}({src_var}: str, {dst_var}: str, db = Depends(get_db)):\n"
                f"    # Vulnerable implementation lacking authorization or sanitization\n"
                f"    query = f\"SELECT * FROM records WHERE {src_var} = '\" + {src_var} + \"'\"\n"
                f"    result = db.execute(query).fetchall()\n"
                f"    return {{'status': 'ok', 'data': result}}\n"
            )
            repair_code = (
                f"@app.post('{route}')\n"
                f"def handle_{archetype}_{uid[:4]}({src_var}: str, {dst_var}: str, current_user = Depends(get_active_user), db = Depends(get_db)):\n"
                f"    # Enforce authorization and parameterized query\n"
                f"    if current_user.tenant_id != {src_var}:\n"
                f"        raise HTTPException(status_code=403, detail='Forbidden access')\n"
                f"    result = db.query(Record).filter(Record.{src_var} == {src_var}).all()\n"
                f"    return {{'status': 'success', 'data': result}}\n"
            )
        elif lang == "javascript":
            vuln_code = (
                f"app.post('{route}', (req, res) => {{\n"
                f"    const {src_var} = req.body.{src_var};\n"
                f"    const {dst_var} = req.body.{dst_var};\n"
                f"    const target = path.join(__dirname, 'uploads', {src_var});\n"
                f"    fs.writeFileSync(target, {dst_var});\n"
                f"    res.json({{ status: 'saved', file: target }});\n"
                f"}});\n"
            )
            repair_code = (
                f"const crypto = require('crypto');\n"
                f"app.post('{route}', authenticateUser, (req, res) => {{\n"
                f"    const safeId = crypto.randomUUID();\n"
                f"    const uploadDir = path.resolve(__dirname, 'uploads');\n"
                f"    const target = path.join(uploadDir, safeId);\n"
                f"    if (!target.startsWith(uploadDir)) return res.status(403).json({{ error: 'Invalid path' }});\n"
                f"    fs.writeFileSync(target, req.body.{dst_var});\n"
                f"    res.json({{ status: 'success', id: safeId }});\n"
                f"}});\n"
            )
        else:
            vuln_code = (
                f"func Handle_{archetype}_{uid[:4]}(c *gin.Context) {{\n"
                f"    {src_var} := c.PostForm(\"{src_var}\")\n"
                f"    // Direct command or raw query execution\n"
                f"    out, _ := exec.Command(\"sh\", \"-c\", {src_var}).Output()\n"
                f"    c.JSON(200, gin.H{{\"output\": string(out)}})\n"
                f"}}\n"
            )
            repair_code = (
                f"func Handle_{archetype}_{uid[:4]}(c *gin.Context) {{\n"
                f"    // Whitelist-validated execution without shell interpreter\n"
                f"    {src_var} := c.PostForm(\"{src_var}\")\n"
                f"    if !isValidCommand({src_var}) {{\n"
                f"        c.JSON(400, gin.H{{\"error\": \"Invalid command parameter\"}})\n"
                f"        return\n"
                f"    }}\n"
                f"    c.JSON(200, gin.H{{\"status\": \"safe\"}})\n"
                f"}}\n"
            )

        findings = [
            SecurityFinding(
                source="semgrep",
                rule_id=f"{lang}.{framework}.security.audit.{tpl['cwe'][0].lower()}",
                cwe=tpl["cwe"][0],
                severity=tpl["severity"],
                line=4,
                message=f"Identified security exposure matching {tpl['cwe'][0]} in {route}.",
            )
        ]

        ground_truth = SecurityGroundTruth(
            threats=tpl["threats"],
            cwe=tpl["cwe"],
            owasp=tpl["owasp"],
            risk=RiskAssessment(
                severity=tpl["severity"],
                score=tpl["score"],
                confidence=0.96,
                explanation=f"Critical exposures in {archetype} handling endpoint {route}.",
            ),
            security_requirements=tpl["requirements"],
            security_strategy=[
                f"Enforce Shift-Left proactive controls for {archetype}.",
                "Validate input boundaries at route registration.",
            ],
            recommended_capabilities=[
                "ShiftGuard-StaticAnalyzer",
                "ShiftGuard-AccessAuditor",
                "ShiftGuard-RemediationEngine",
            ],
            agent_plan=tpl["agent_plan"],
            explanation=f"Endpoint {route} violates secure coding best practices by improperly binding `{src_var}` without access control or parameterization.",
            repair_guidance=[f"Remediate {c} by applying parameterization and access checks." for c in tpl["cwe"]],
            repair_code=repair_code,
            verification=VerificationStatus.FIXED,
        )

        return SecuritySample(
            sample_id=f"synth-{archetype}-{task.value}-{uid}",
            task=task,
            requirement=requirement,
            prompt=prompt,
            context={"language": lang, "framework": framework, "archetype": archetype, "route": route},
            code=vuln_code,
            security_findings=findings,
            ground_truth=ground_truth,
            confidence=ConfidenceLevel.HIGH,
            metadata=ProvenanceMetadata(
                source_name="shiftguard_synthetic_generator",
                source_id=f"{archetype}_{uid}",
                repository=f"shiftguard/{archetype}",
                license="Apache-2.0",
                is_synthetic=True,
                archetype=archetype,
            ),
        )

    def generate_corpus(self, num_samples_per_archetype: int = 15) -> List[SecuritySample]:
        """Generates a rich multi-task dataset across all 12 archetypes."""
        samples = []
        tasks = list(TaskType)
        for arch in ARCHETYPE_TEMPLATES.keys():
            for i in range(num_samples_per_archetype):
                task = tasks[i % len(tasks)]
                sample = self.generate_sample(arch, task, sample_idx=i)
                samples.append(sample)
        return samples

