"""Open the Streamlit Community Cloud app and wake it if it is asleep.

Community Cloud puts apps to sleep after 12 hours without visitors; a
scheduled visit from a real browser keeps it running. Exits non-zero when
the app does not come up, so the scheduled run is marked as failed."""
import os
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://xrdkit.streamlit.app/"
WAKE = "Yes, get this app back up"


def app_frame(page):
    for f in page.frames:
        try:
            if f.locator('[data-testid="stApp"]').count():
                return f
        except Exception:
            pass
    return None


with sync_playwright() as p:
    browser = p.chromium.launch(channel=os.environ.get("BROWSER_CHANNEL") or None)
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_timeout(10_000)
    btn = page.get_by_role("button", name=WAKE)
    if btn.count():
        print("app was asleep: waking it")
        btn.first.click()
    else:
        print("app was awake")
    for _ in range(60):                     # up to 10 min for a cold start
        f = app_frame(page)
        if f is not None and f.locator('[data-testid="stSidebar"]').count():
            print("app is up")
            browser.close()
            sys.exit(0)
        page.wait_for_timeout(10_000)
    page.screenshot(path="keep_awake_failure.png", full_page=True)
    print("app did not come up within 10 minutes", file=sys.stderr)
    browser.close()
    sys.exit(1)
