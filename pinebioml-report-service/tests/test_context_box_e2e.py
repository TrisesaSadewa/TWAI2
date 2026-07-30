import uuid
import pytest
from fastapi.testclient import TestClient
from core.main import app
from core.security import generate_csrf_token

client = TestClient(app)

def test_access_page_renders_context_boxes():
    """Verify that access.html renders the optional additional study context box and template button."""
    response = client.get("/Statistical_Analysis/upload")
    assert response.status_code == 200
    content = response.text
    
    assert "Additional Study Context & Metadata" in content
    assert "Optional — Complements Auto-Detection" in content
    assert "insert-template-btn" in content
    assert "name=\"additional_context\"" in content
    assert "context-textarea" in content

def test_upload_check_passes_additional_context_to_check_page():
    """Verify that posting additional_context from access.html forwards it to check.html."""
    sample_context = "Disease: Breast Cancer Stage II. Purpose: Early diagnostic screening."
    valid_uuid = str(uuid.uuid4())
    token = generate_csrf_token()
    
    # Submit test dataset form with CSRF cookie & form token
    response = client.post(
        f"/Statistical_Analysis/upload/check_test/{valid_uuid}/",
        data={
            "dataset": "pd",
            "additional_context": sample_context,
            "csrf_token": token,
            "csrfmiddlewaretoken": token
        },
        cookies={"pinebioml_csrf": token},
        follow_redirects=True
    )
    assert response.status_code == 200
    content = response.text
    
    # Ensure check.html receives and renders hidden input with additional_context
    assert 'name="additional_context"' in content
    assert sample_context in content

def test_check_passes_additional_context_to_setting_page():
    """Verify that posting target_column and additional_context from check.html forwards it to setting.html."""
    sample_context = "Disease: Heart Disease Cleveland. Goal: Risk classification."
    valid_uuid = str(uuid.uuid4())
    token = generate_csrf_token()
    
    # Submit target selection with valid CSRF cookie & form token
    response = client.post(
        f"/Statistical_Analysis/setting/{valid_uuid}/",
        data={
            "target_column": "target",
            "additional_context": sample_context,
            "csrf_token": token,
            "csrfmiddlewaretoken": token
        },
        cookies={"pinebioml_csrf": token},
        follow_redirects=True
    )
    assert response.status_code == 200
    content = response.text
    
    # Ensure setting.html receives and renders hidden input with additional_context
    assert 'name="additional_context"' in content
    assert sample_context in content
