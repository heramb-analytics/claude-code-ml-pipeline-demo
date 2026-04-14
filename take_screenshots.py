"""Take 6 Playwright screenshots of the running API dashboard."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = Path("reports/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://localhost:8000"

shots = [
    ("01_dashboard_home.png",     BASE_URL + "/",          None),
    ("02_form_filled.png",        BASE_URL + "/",          "fill_form"),
    ("03_prediction_result.png",  BASE_URL + "/",          "submit"),
    ("04_swagger_docs.png",       BASE_URL + "/docs",      None),
    ("05_metrics_endpoint.png",   BASE_URL + "/metrics",   None),
    ("06_health_endpoint.png",    BASE_URL + "/health",    None),
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    for i, (filename, url, action) in enumerate(shots, 1):
        print(f"   📸 Taking screenshot {i}/6: {filename}...")
        page.goto(url, wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(800)

        if action == "fill_form":
            page.fill('input[name="amount"]', "25000.00")
            page.select_option('select[name="merchant_category"]', "electronics")
            page.select_option('select[name="card_type"]', "prepaid")
            page.fill('input[name="transaction_hour"]', "2")
            page.fill('input[name="is_international"]', "1")

        elif action == "submit":
            page.fill('input[name="amount"]', "25000.00")
            page.select_option('select[name="merchant_category"]', "electronics")
            page.select_option('select[name="card_type"]', "prepaid")
            page.fill('input[name="transaction_hour"]', "2")
            page.fill('input[name="is_international"]', "1")
            page.click('button[type="submit"]')
            page.wait_for_selector("#result:not(.hidden)", timeout=5000)
            page.wait_for_timeout(500)

        out_path = SCREENSHOTS_DIR / filename
        page.screenshot(path=str(out_path), full_page=True)
        size_kb = out_path.stat().st_size // 1024
        print(f"   ✅ Saved: reports/screenshots/{filename} ({size_kb}KB)")

    browser.close()

print("   📸 All 6 screenshots captured.")
