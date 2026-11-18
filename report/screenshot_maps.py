"""Render Folium HTML maps to PNG for the PDF report (headless Chrome).

Folium pages load Leaflet & friends from CDNs. When the machine is offline
(or a CDN is blocked) the script serves those libraries from a local
node_modules folder instead (`npm install leaflet leaflet.awesome-markers
jquery bootstrap@5`). Basemap tiles that cannot load are simply left blank;
the road network layer drawn from .nodes/.arcs still gives context.

usage: python report/screenshot_maps.py <node_modules_dir> <out_dir> map1.html [map2.html ...]
"""
import os
import sys

from playwright.sync_api import sync_playwright

NM = sys.argv[1]
OUT = sys.argv[2]
LOCAL = {
    "leaflet.js": "leaflet/dist/leaflet.js",
    "leaflet.css": "leaflet/dist/leaflet.css",
    "leaflet.awesome-markers.js": "leaflet.awesome-markers/dist/leaflet.awesome-markers.js",
    "leaflet.awesome-markers.css": "leaflet.awesome-markers/dist/leaflet.awesome-markers.css",
    "jquery.min.js": "jquery/dist/jquery.min.js",
    "jquery-3.7.1.min.js": "jquery/dist/jquery.min.js",
    "bootstrap.bundle.min.js": "bootstrap/dist/js/bootstrap.bundle.min.js",
    "bootstrap.min.css": "bootstrap/dist/css/bootstrap.min.css",
}
os.makedirs(OUT, exist_ok=True)


def route(r):
    url = r.request.url
    if url.startswith("file://"):
        return r.continue_()
    name = url.split("?")[0].rsplit("/", 1)[-1]
    if name in LOCAL:
        path = os.path.join(NM, LOCAL[name])
        ctype = "text/css" if name.endswith(".css") else "application/javascript"
        return r.fulfill(path=path, content_type=ctype)
    try:
        return r.continue_()
    except Exception:
        return r.abort()


with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=1.5)
    pg.route("**/*", route)
    for f in sys.argv[3:]:
        pg.goto("file://" + os.path.abspath(f), wait_until="domcontentloaded")
        pg.wait_for_timeout(2500)
        out = os.path.join(OUT, os.path.basename(f).replace(".html", ".png"))
        pg.screenshot(path=out)
        print("saved", out)
    b.close()
