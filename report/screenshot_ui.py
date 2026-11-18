"""Capture screenshots of the evaluator UI talking to a running API.

Assumes: API on http://localhost:8000 and the UI served on http://localhost:8080
usage: python report/screenshot_ui.py <out_dir> <trip_id> [<trip_id> ...]
"""
import os
import sys

from playwright.sync_api import sync_playwright

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=1.25)
    pg.goto("http://localhost:8080/?api=http://localhost:8000")
    pg.wait_for_selector(".trip")
    pg.screenshot(path=os.path.join(out, "ui_trip_list.png"))
    for tid in sys.argv[2:]:
        pg.click(f'.trip[data-id="{tid}"]')
        pg.wait_for_selector(".kpi .v", timeout=120_000)
        pg.wait_for_timeout(1200)
        pg.screenshot(path=os.path.join(out, f"ui_result_{tid}.png"))
        print("saved", tid)
    b.close()
