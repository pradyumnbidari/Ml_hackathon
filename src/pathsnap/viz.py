"""Folium map builders shared by EDA, HMM analysis and the report."""
from __future__ import annotations

import folium
import numpy as np


def _bbox_arcs(net, lonlat, margin_m=150.0):
    xy = net.proj.fwd(lonlat)
    lo, hi = xy.min(0) - margin_m, xy.max(0) + margin_m
    inside = ((net.mid >= lo) & (net.mid <= hi)).all(axis=1)
    return np.where(inside)[0]


def _arc_lines(net, arc_ids):
    A = net.nodes_ll[net.arcs[arc_ids, 0]]
    B = net.nodes_ll[net.arcs[arc_ids, 1]]
    return [[[a[1], a[0]], [b[1], b[0]]] for a, b in zip(A, B)]


def _chain(net, arc_ids):
    """Continuous lat/lon polyline segments for an ordered arc list."""
    segs, cur = [], []
    for i, ai in enumerate(arc_ids):
        f, t = net.arcs[ai]
        if cur and (i == 0 or net.arcs[arc_ids[i - 1], 1] != f):
            segs.append(cur)
            cur = []
        if not cur:
            cur.append([net.nodes_ll[f, 1], net.nodes_ll[f, 0]])
        cur.append([net.nodes_ll[t, 1], net.nodes_ll[t, 0]])
    if cur:
        segs.append(cur)
    return segs


def trip_map(net, track_lonlat, reference_arcs=None, predicted_arcs=None,
             classifier_arcs=None, title="", max_points=4000, show_network=True):
    lat, lon = track_lonlat[:, 1], track_lonlat[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], tiles="cartodbpositron",
                   control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    if show_network:
        fg = folium.FeatureGroup(name="Road network (nearby arcs)", show=True)
        ids = _bbox_arcs(net, track_lonlat)
        folium.PolyLine(_arc_lines(net, ids), color="#9e9e9e", weight=1.5,
                        opacity=0.6).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup(name="Raw GPS observations")
    step = max(1, len(track_lonlat) // max_points)
    for lo, la in track_lonlat[::step]:
        folium.CircleMarker([la, lo], radius=1.6, color="#1f4e9e", fill=True,
                            fill_opacity=0.55, weight=0).add_to(fg)
    fg.add_to(m)
    if reference_arcs is not None:
        fg = folium.FeatureGroup(name="Reference route (.route)")
        for seg in _chain(net, list(reference_arcs)):
            folium.PolyLine(seg, color="#1b9e3e", weight=6, opacity=0.75).add_to(fg)
        fg.add_to(m)
    if classifier_arcs is not None:
        fg = folium.FeatureGroup(name="Classifier-only prediction", show=False)
        for seg in _chain(net, list(classifier_arcs)):
            folium.PolyLine(seg, color="#ff7f0e", weight=3, opacity=0.9,
                            dash_array="6,6").add_to(fg)
        fg.add_to(m)
    if predicted_arcs is not None:
        fg = folium.FeatureGroup(name="Predicted route (classifier + HMM)")
        for seg in _chain(net, list(predicted_arcs)):
            folium.PolyLine(seg, color="#d62728", weight=3.5, opacity=0.95).add_to(fg)
        fg.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;
    padding:8px 12px;border:1px solid #999;border-radius:6px;font:12px sans-serif">
    <b>{title}</b><br>
    <span style="color:#1f4e9e">&#9679;</span> Raw GPS &nbsp;
    <span style="color:#9e9e9e">&#9473;</span> Road network<br>
    <span style="color:#1b9e3e">&#9473;</span> Reference route &nbsp;
    <span style="color:#d62728">&#9473;</span> Predicted (HMM) &nbsp;
    <span style="color:#ff7f0e">&#9476;</span> Classifier-only
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m
