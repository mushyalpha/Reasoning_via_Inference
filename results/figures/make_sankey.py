import plotly.graph_objects as go

# ---- Data (from the source figure) --------------------------------------
# node index : (label, count, pct_of_grid)
nodes = [
    ("All trials",              7560, 100.0),  # 0
    ("Proposals",                7149,  94.6),  # 1
    ("No CGN proposal",           411,   5.4),  # 2
    ("Pre-grasp collision",      5501,  72.8),  # 3
    ("Gate-pass",                1648,  21.8),  # 4
    ("Success",                   380,   5.0),  # 5
    ("Closed without lifting",    784,  10.4),  # 6
    ("Object left footprint",     484,   6.4),  # 7
]

links = [
    (0, 1, 7149),
    (0, 2,  411),
    (1, 3, 5501),
    (1, 4, 1648),
    (4, 5,  380),
    (4, 6,  784),
    (4, 7,  484),
]

# ---- Colour palette (Okabe–Ito, colour-blind safe) -----------------------
node_color = {
    0: "#4D4D4D",  # all trials (root)
    1: "#0072B2",  # proposals (on main path)
    2: "#D55E00",  # no CGN proposal (dropout)
    3: "#8C8C8C",  # pre-grasp collision (dominant failure mode)
    4: "#0072B2",  # gate-pass (on main path)
    5: "#009E73",  # success (terminal, positive)
    6: "#E69F00",  # closed without lifting (terminal failure)
    7: "#D55E00",  # object left footprint (terminal failure)
}

def hex_to_rgba(h, alpha):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

link_colors = [hex_to_rgba(node_color[t], 0.45) for _, t, _ in links]

# ---- Manual layout (mirrors the original left-to-right box layout) -------
node_x = [0.001, 0.32, 0.32, 0.64, 0.64, 0.999, 0.999, 0.999]
node_y = [0.45,  0.22, 0.86, 0.68, 0.18, 0.05,  0.32,  0.64]

labels = [
    f"{name}<br>{count:,}  ({pct:.1f}%)" for name, count, pct in nodes
]

fig = go.Figure(data=[go.Sankey(
    arrangement="fixed",
    node=dict(
        pad=28,
        thickness=20,
        line=dict(color="black", width=0.6),
        label=labels,
        color=[node_color[i] for i in range(len(nodes))],
        x=node_x,
        y=node_y,
    ),
    link=dict(
        source=[s for s, t, v in links],
        target=[t for s, t, v in links],
        value=[v for s, t, v in links],
        color=link_colors,
    ),
    textfont=dict(family="Times New Roman, Georgia, serif", size=15, color="black"),
)])

title_text = (
    "Trial flow of the 7,560-trial confirmatory grid<br>"
    "<span style='font-size:13px;color:#333333'>"
    "P(Y=1) = P(N&gt;0) \u00b7 P(G=1 | N&gt;0) \u00b7 P(Y=1 | G=1)"
    "&nbsp;&nbsp;&nbsp;(terminal-node counts are shares of all 7,560 trials)"
    "</span>"
)

fig.update_layout(
    title=dict(
        text=title_text,
        font=dict(family="Times New Roman, Georgia, serif", size=20, color="black"),
        x=0.01,
        xanchor="left",
        y=0.98,
        yanchor="top",
    ),
    font=dict(family="Times New Roman, Georgia, serif", size=14, color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=110, b=50),
    width=1150,
    height=680,
)

fig.write_image("/mnt/user-data/outputs/trial_flow_sankey.pdf", scale=1)
fig.write_image("/mnt/user-data/outputs/trial_flow_sankey.svg", scale=1)
fig.write_image("/mnt/user-data/outputs/trial_flow_sankey.png", scale=3)
print("done")
