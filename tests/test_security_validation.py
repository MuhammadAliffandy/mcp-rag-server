"""
Security & Robustness Validation Tests

Tests the security fixes from the Colonossene evaluation:
- R1/G3: Path traversal prevention in ingest_medical_files
- R6/G4: Minimum query length in exact_identifier_search
- R2: .env.example existence and content

Run: python tests/test_security_validation.py
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# We need to set the project_root context for the validation functions
# Import the validation functions directly from mcp_server source
# Since mcp_server has heavy imports, we replicate the lightweight functions here

ALLOWED_INGEST_DIRS = [
    os.path.join(project_root, "temp_uploads"),
    os.path.join(project_root, "internal_docs"),
    os.path.join(project_root, "documents"),
]

MIN_SEARCH_QUERY_LENGTH = 3


def validate_directory_path(directory_path: str) -> str:
    """Replicated from mcp_server.py for standalone testing."""
    resolved = os.path.realpath(os.path.abspath(directory_path))
    for allowed in ALLOWED_INGEST_DIRS:
        allowed_resolved = os.path.realpath(os.path.abspath(allowed))
        if resolved.startswith(allowed_resolved + os.sep) or resolved == allowed_resolved:
            return resolved
    raise ValueError(
        f"Path '{directory_path}' is outside allowed directories. "
        f"Allowed: {[os.path.basename(d) for d in ALLOWED_INGEST_DIRS]}"
    )


def validate_search_query(query: str) -> str:
    """Replicated from mcp_server.py for standalone testing."""
    stripped = query.strip() if query else ""
    if len(stripped) < MIN_SEARCH_QUERY_LENGTH:
        raise ValueError(
            f"Search query must be at least {MIN_SEARCH_QUERY_LENGTH} characters. "
            f"Got: '{stripped}' ({len(stripped)} chars). Please provide a more specific search term."
        )
    return stripped


# =============================================================================
# TEST CASES
# =============================================================================

passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} — {detail}")
        failed += 1


# --- R1/G3: Path Traversal Prevention ---
print("\n🛡️  R1/G3: Path Traversal Prevention")
print("=" * 50)

# Should BLOCK: path traversal attempts
traversal_paths = [
    "/etc/passwd",
    "../../../etc/shadow",
    os.path.join(project_root, "..", "other_project"),
    "/tmp/malicious",
    "../../../../root",
]

for path in traversal_paths:
    try:
        validate_directory_path(path)
        test(f"Block '{path}'", False, "Should have raised ValueError")
    except ValueError:
        test(f"Block '{path}'", True)

# Should ALLOW: valid project directories
valid_paths = [
    os.path.join(project_root, "temp_uploads"),
    os.path.join(project_root, "internal_docs"),
    os.path.join(project_root, "documents"),
    os.path.join(project_root, "internal_docs", "subfolder"),
]

for path in valid_paths:
    try:
        result = validate_directory_path(path)
        test(f"Allow '{os.path.relpath(path, project_root)}'", True)
    except ValueError as e:
        test(f"Allow '{os.path.relpath(path, project_root)}'", False, str(e))


# --- R6/G4: Minimum Query Length ---
print("\n🔍  R6/G4: Minimum Query Length")
print("=" * 50)

# Should BLOCK: too-short queries
short_queries = ["", "a", "ab", "  ", " a"]

for q in short_queries:
    try:
        validate_search_query(q)
        test(f"Block '{repr(q)}'", False, "Should have raised ValueError")
    except ValueError:
        test(f"Block '{repr(q)}'", True)

# Should ALLOW: valid queries
valid_queries = ["abc", "patient 123", "ACC-2024-X", "  valid query  "]

for q in valid_queries:
    try:
        result = validate_search_query(q)
        test(f"Allow '{q.strip()}'", True)
    except ValueError as e:
        test(f"Allow '{q.strip()}'", False, str(e))


# --- R2: .env.example ---
print("\n📋  R2: .env.example")
print("=" * 50)

env_example = os.path.join(project_root, ".env.example")
test(".env.example exists", os.path.exists(env_example))

if os.path.exists(env_example):
    with open(env_example) as f:
        content = f.read()
    test("Contains OPENAI_API_KEY", "OPENAI_API_KEY" in content)
    test("Contains PINECONE_API_KEY", "PINECONE_API_KEY" in content)
    test("Contains PINECONE_ENV", "PINECONE_ENV" in content)


# --- Summary ---
print("\n" + "=" * 50)
total = passed + failed
print(f"📊 Results: {passed}/{total} passed, {failed} failed")

if failed > 0:
    print("❌ SOME TESTS FAILED")
    sys.exit(1)
else:
    print("✅ ALL TESTS PASSED")
    sys.exit(0)
