from cv_agent.security import redact_sensitive


def test_redact_sensitive_removes_keys_email_and_phone():
    text = (
        "token=private-token-value "
        "AIza-THIS-IS-A-PRIVATE-KEY "
        "person@example.org +49 176 12345678"
    )
    redacted = redact_sensitive(text)
    assert "private-token-value" not in redacted
    assert "AIza-THIS" not in redacted
    assert "person@example.org" not in redacted
    assert "12345678" not in redacted


def test_redact_sensitive_removes_known_nonstandard_secret():
    assert redact_sensitive("failure: xyz-secret", ["xyz-secret"]) == (
        "failure: [REDACTED]"
    )
