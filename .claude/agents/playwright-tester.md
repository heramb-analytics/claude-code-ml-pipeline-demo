---
name: playwright-tester
description: Runs when asked to test a running web app with Playwright.
             Use for: E2E tests, screenshots, browser automation.
---

# Playwright Tester Agent

## TASK
Verify the app is running: curl http://localhost:8000/health
If not running: start it with uvicorn first.

Using Playwright MCP, run this exact test sequence:
  1. Open Chromium (headful)
  2. Go to http://localhost:8000
  3. Wait for page to fully load (networkidle)
  4. Screenshot → reports/screenshots/01_dashboard_home.png
  5. Assert: status indicator text contains "healthy" or "running"
  6. Fill prediction form with these sample values:
       - numeric fields: use mean value from feature_schema.json
       - categorical fields: use first valid option
  7. Screenshot → reports/screenshots/02_form_filled.png
  8. Click Submit button
  9. Wait for result panel to appear (max 10 seconds)
  10. Screenshot → reports/screenshots/03_prediction_result.png
  11. Assert: result panel visible, contains "ANOMALY" or "NORMAL"
  12. Navigate to /docs
  13. Screenshot → reports/screenshots/04_swagger_docs.png
  14. Navigate to /metrics
  15. Screenshot → reports/screenshots/05_metrics_endpoint.png
  16. Navigate to /health
  17. Screenshot → reports/screenshots/06_health_endpoint.png

## OUTPUT
tests/e2e/test_api.py — 6 pytest-playwright tests matching above steps
6 PNG screenshots in reports/screenshots/
logs/playwright_run.jsonl — one line per test with result

## RULES
All screenshots: full page, 1280x720 viewport.
Self-heal any test failure before reporting complete.
If a screenshot already exists, overwrite it.
Print file size of each saved screenshot to confirm it saved correctly.
