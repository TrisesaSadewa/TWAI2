import os
import sys

# Ensure the app can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from core.main import app

client = TestClient(app, raise_server_exceptions=False)

print("Starting Security Verification...\n")

# 1. Test XSS payloads in Report IDs
print("1. Testing XSS in Report IDs (Expect 400 Bad Request):")
xss_payload = "<script>alert(1)"
response = client.get(f"/report/status/{xss_payload}")
print(f"Status Code: {response.status_code}")
assert response.status_code == 400, "XSS Payload was not rejected!"
print("[OK] XSS payload rejected.")

# 2. Test Path Traversal
print("\n2. Testing Path Traversal on artifacts (Expect 400 Bad Request):")
from core.security import safe_path_join
from fastapi import HTTPException
try:
    safe_path_join("/tmp/base", "../../../etc/passwd")
    print("[FAIL] Path traversal was NOT rejected!")
    sys.exit(1)
except HTTPException as e:
    assert e.status_code == 400
    print("[OK] Path traversal blocked.")

# 3. Test Unauthorized Access
print("\n3. Testing Unauthorized Access (Missing API Key):")
# The /report/{report_id} now requires api_key
valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
response = client.get(f"/report/{valid_uuid}")
print(f"Status Code: {response.status_code}")
assert response.status_code == 401, f"API Key was not enforced! Status: {response.status_code}"
print("[OK] Missing API key blocked.")

# 4. Test Security Headers
print("\n4. Testing Security Headers on UI route:")
response = client.get(f"/Statistical_Analysis/download/{valid_uuid}/")
print(f"Headers: {response.headers}")
assert "Content-Security-Policy" in response.headers, "CSP header missing!"
assert "X-Content-Type-Options" in response.headers, "X-Content-Type-Options missing!"
assert "X-Frame-Options" in response.headers, "X-Frame-Options missing!"
# Check if nonce is injected in HTML
if "nonce=" in response.text:
    print("[OK] CSP Nonce found in HTML.")
else:
    print("[FAIL] CSP Nonce missing in HTML.")
print("[OK] Security Headers present.")

# 5. Test Rate Limiting
print("\n5. Testing Rate Limiting (Expect 429 Too Many Requests after limit):")
# The /report/{report_id}/clone is limited to 5/minute
headers = {"X-API-Key": "7c824c8b25d14e03b3d2f954f9a567d2_new_key"}
# We might get 404 because report not found, but rate limiter happens before route logic
for i in range(6):
    res = client.post(f"/report/{valid_uuid}/clone", headers=headers)
    if res.status_code == 429:
        print(f"Hit rate limit on attempt {i+1}")
        break
else:
    print("[FAIL] Rate limiting failed to trigger!")
    sys.exit(1)
# 6. Test CSRF on UI Forms (Expect 403 Forbidden)
print("\n6. Testing CSRF Protection on UI POST (Expect 403 Forbidden):")
# /Statistical_Analysis/setting/{uuid}/ is a UI route that accepts POST. Without token, it should fail.
response = client.post(f"/Statistical_Analysis/setting/{valid_uuid}/", data={"target_column": "test"})
print(f"Status Code: {response.status_code}")
assert response.status_code == 403, "CSRF was not enforced on UI POST!"
print("[OK] CSRF Protection active on UI forms.")

# 7. Test XSS in Filenames (Expect sanitized output)
print("\n7. Testing XSS payload in filename sanitization:")
from core.security import sanitize_filename
dirty_filename = 'data"<script>alert(1)</script>.csv'
clean = sanitize_filename(dirty_filename)
print(f"Cleaned filename: {clean}")
assert "<script>" not in clean, "Filename was not sanitized!"
print("[OK] XSS in filenames sanitized.")

print("\n[SUCCESS] All Security Verifications Passed!")
