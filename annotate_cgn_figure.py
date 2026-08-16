"""
annotate_cgn_figure.py
======================
Creates a zoomed-in, fully annotated figure of the CGN grasp distribution
for use in the MSc report and supervisor presentation.

Run:
    python annotate_cgn_figure.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D

# ── load saved predictions ──────────────────────────────────────────────────
_PROJECT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(_PROJECT,
      'results/figures/cgn_predictions_phi45_theta0_sd0.0_rho1.0.npz')
OUT_DIR = os.path.join(_PROJECT, 'results/figures')
os.makedirs(OUT_DIR, exist_ok=True)

data      = np.load(NPZ, allow_pickle=True)
pc_full   = data['pc_full']                         # (N,3) camera-frame
pred_grasps = data['pred_grasps_cam'].item()        # {1: (133,4,4)}
scores    = data['scores'].item()                   # {1: (133,)}

# Flatten grasps & scores
all_pos, all_sc = [], []
for k in pred_grasps:
    for g, s in zip(pred_grasps[k], scores[k]):
        all_pos.append(g[:3, 3])
        all_sc.append(float(s))
all_pos = np.array(all_pos)   # (133,3)  grasp contact positions
all_sc  = np.array(all_sc)    # (133,)

best_idx = int(np.argmax(all_sc))
best_pos = all_pos[best_idx]

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Wide overview with structural labels
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10), facecolor='#0d1117')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0d1117')

# --- Point cloud: colour-code different scene regions by Z depth ----
# In camera frame (Z = distance from camera):
#   Z ~ 0.6-0.7  = cylinder (close, small blob)
#   Z ~ 0.8-0.9  = table surface (large flat layer)
#   Z ~ 1.0-1.3  = robot arm body (mid-distance)
#   Z ~ 1.4-1.6  = robot arm upper joints (far)

step = max(1, len(pc_full) // 4000)
pc_s = pc_full[::step]
z = pc_s[:, 2]

# Cylinder points (closest to camera, near Z=0.65 in cam frame)
mask_cyl   = z < 0.75
mask_table = (z >= 0.75) & (z < 1.05)
mask_arm   = z >= 1.05

ax.scatter(pc_s[mask_cyl,0],  pc_s[mask_cyl,1],  pc_s[mask_cyl,2],
           c='#e06c75', s=6, alpha=0.7, label='Target cylinder')
ax.scatter(pc_s[mask_table,0],pc_s[mask_table,1],pc_s[mask_table,2],
           c='#61afef', s=2, alpha=0.25, label='Table surface')
ax.scatter(pc_s[mask_arm,0],  pc_s[mask_arm,1],  pc_s[mask_arm,2],
           c='#98c379', s=2, alpha=0.2,  label='Robot arm body')

# Grasp proposals coloured by confidence
cmap = matplotlib.colormaps['plasma']
mn, mx = all_sc.min(), all_sc.max()
cols = [cmap((s - mn) / (mx - mn)) for s in all_sc]
sc = ax.scatter(all_pos[:,0], all_pos[:,1], all_pos[:,2],
                c=all_sc, cmap='plasma', s=40, alpha=0.9, zorder=5,
                vmin=mn, vmax=mx, label='Grasp proposals (133)')
ax.scatter(*best_pos, c='red', s=200, marker='*', zorder=10,
           label=f'Best grasp (score={all_sc[best_idx]:.4f})')

cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.12)
cbar.set_label('CGN Confidence Score', color='white', fontsize=11)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# Axis labels
ax.set_xlabel('X (m) - horizontal', color='#abb2bf', fontsize=9, labelpad=8)
ax.set_ylabel('Y (m) - vertical\n(+Y = down in cam)', color='#abb2bf', fontsize=9, labelpad=8)
ax.set_zlabel('Z (m) - depth\n(= distance from camera)', color='#abb2bf', fontsize=9, labelpad=8)
ax.tick_params(colors='#666677')

ax.set_title('Contact-GraspNet: All 133 Grasp Proposals on Scene Point Cloud\n'
             '(Camera frame, φ=45°, θ=0°, σ_d=0.0, ρ=1.0 — Baseline clean conditions)',
             color='white', fontsize=12, fontweight='bold', pad=15)

leg = ax.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white',
                fontsize=9, framealpha=0.85)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'cgn_annotated_overview.png')
plt.savefig(out1, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f'Saved: {out1}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: ZOOMED IN on the croissant cluster with arrows + annotations
# ─────────────────────────────────────────────────────────────────────────────

# Zoom bounds: tightly around the grasps + cylinder
x_min, x_max = all_pos[:,0].min()-0.05, all_pos[:,0].max()+0.05
y_min, y_max = all_pos[:,1].min()-0.05, all_pos[:,1].max()+0.05
z_min, z_max = all_pos[:,2].min()-0.05, all_pos[:,2].max()+0.05

# Also include cylinder points in zoom
cx_mask = (pc_full[:,0] > x_min-0.05) & (pc_full[:,0] < x_max+0.05) & \
          (pc_full[:,1] > y_min-0.05) & (pc_full[:,1] < y_max+0.05) & \
          (pc_full[:,2] > z_min-0.05) & (pc_full[:,2] < z_max+0.05)
pc_zoom = pc_full[cx_mask]

fig2 = plt.figure(figsize=(16, 11), facecolor='#0d1117')
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_facecolor('#0d1117')

# Cylinder point cloud in zoom
if len(pc_zoom) > 0:
    step2 = max(1, len(pc_zoom) // 2000)
    pz = pc_zoom[::step2]
    ax2.scatter(pz[:,0], pz[:,1], pz[:,2],
                c='#e06c75', s=12, alpha=0.55, label='Cylinder surface (point cloud)')

# All grasps
sc2 = ax2.scatter(all_pos[:,0], all_pos[:,1], all_pos[:,2],
                  c=all_sc, cmap='plasma', s=80, alpha=0.95, zorder=6,
                  vmin=mn, vmax=mx)
ax2.scatter(*best_pos, c='red', s=400, marker='*', zorder=10,
            label=f'Best grasp  score={all_sc[best_idx]:.4f}')

# Draw approach vectors for top-5 grasps (Z-axis of each grasp = approach dir)
top5_idx = np.argsort(all_sc)[::-1][:5]
for k in pred_grasps:
    grasps_k = pred_grasps[k]
    sc_k     = scores[k]
    order = np.argsort(sc_k)[::-1][:5]
    for rank, gi in enumerate(order):
        g   = grasps_k[gi]
        pos = g[:3, 3]
        # approach direction = gripper Z axis = g[:3,2]
        approach = g[:3, 2] * 0.06   # scale for visibility
        ax2.quiver(pos[0], pos[1], pos[2],
                   approach[0], approach[1], approach[2],
                   length=1.0, normalize=False,
                   color='yellow' if rank == 0 else 'cyan',
                   linewidth=2.5 if rank == 0 else 1.5,
                   arrow_length_ratio=0.4)

cbar2 = fig2.colorbar(sc2, ax=ax2, shrink=0.5, pad=0.12)
cbar2.set_label('CGN Confidence Score', color='white', fontsize=11)
cbar2.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')

ax2.set_xlim(x_min-0.03, x_max+0.03)
ax2.set_ylim(y_min-0.03, y_max+0.03)
ax2.set_zlim(z_min-0.03, z_max+0.03)

ax2.set_xlabel('X (m)', color='#abb2bf', fontsize=10)
ax2.set_ylabel('Y (m)', color='#abb2bf', fontsize=10)
ax2.set_zlabel('Z (m) depth', color='#abb2bf', fontsize=10)
ax2.tick_params(colors='#777788')

ax2.set_title(
    'ZOOMED: CGN Grasp Proposals on Cylinder Surface\n'
    'Each dot = one 6-DoF grasp candidate   |   Arrows = top-5 approach directions\n'
    'Yellow arrow = best grasp   |   Colour = confidence (yellow=high, purple=low)',
    color='white', fontsize=12, fontweight='bold', pad=15)

leg2 = ax2.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white',
                  fontsize=10, framealpha=0.9)

# Add text box explaining the arc shape
textstr = (
    'Why arc-shaped?\n'
    'CGN proposes grasps from\n'
    'MULTIPLE approach directions\n'
    'around the cylinder. The arc\n'
    'traces valid contact points on\n'
    'the visible cylinder surface.'
)
props = dict(boxstyle='round,pad=0.6', facecolor='#1f2937',
             edgecolor='#4b9cd3', alpha=0.92)
ax2.text2D(0.73, 0.88, textstr, transform=ax2.transAxes,
           fontsize=9, color='white', verticalalignment='top',
           bbox=props)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'cgn_zoomed_annotated.png')
plt.savefig(out2, dpi=160, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f'Saved: {out2}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Top-down 2D view (XY plane) - clearest view of the arc
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
})
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')

## Left: top-down XY view
ax3L = axes3[0]
ax3L.set_facecolor('white')

# Cylinder cloud top-down
if len(pc_zoom) > 0:
    ax3L.scatter(pc_zoom[:,0], pc_zoom[:,1], c='#8a8a8a', s=4, alpha=0.35,
                 label='Cylinder surface')

sc3 = ax3L.scatter(all_pos[:,0], all_pos[:,1], c=all_sc, cmap='plasma',
                    s=60, alpha=0.9, vmin=mn, vmax=mx, zorder=5,
                    label='Grasp proposals')
ax3L.scatter(best_pos[0], best_pos[1], c='red', s=300, marker='*', zorder=10,
             label=f'Best grasp\n(score={all_sc[best_idx]:.4f})')

cbar3 = fig3.colorbar(sc3, ax=ax3L, shrink=0.8)
cbar3.set_label('Confidence', color='black', fontsize=10)
cbar3.ax.yaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='black')

ax3L.set_xlabel('X (m)', color='black', fontsize=11)
ax3L.set_ylabel('Y (m)', color='black', fontsize=11)
ax3L.set_title('(a) Top-down projection (XY)', loc='left',
               color='black', fontsize=12)
ax3L.tick_params(colors='black')
ax3L.spines[:].set_color('#777777')
ax3L.legend(facecolor='white', labelcolor='black', edgecolor='#777777',
            framealpha=1.0, fontsize=9)

# Annotation callouts
mid_x = all_pos[:,0].mean()
mid_y = all_pos[:,1].mean()
ax3L.annotate('The crescent = 133 grasp\ncandidate contact points,\nall on the cylinder surface',
               xy=(mid_x, mid_y), xytext=(mid_x + 0.15, mid_y + 0.12),
               color='black', fontsize=9, ha='left',
               arrowprops=dict(arrowstyle='->', color='#4c78a8', lw=1.2),
               bbox=dict(boxstyle='round,pad=0.4', fc='white',
                         ec='#777777', alpha=0.95))

## Right: side view (XZ = depth profile)
ax3R = axes3[1]
ax3R.set_facecolor('white')

if len(pc_zoom) > 0:
    ax3R.scatter(pc_zoom[:,0], pc_zoom[:,2], c='#8a8a8a', s=4, alpha=0.35)
sc3R = ax3R.scatter(all_pos[:,0], all_pos[:,2], c=all_sc, cmap='plasma',
                     s=60, alpha=0.9, vmin=mn, vmax=mx, zorder=5)
ax3R.scatter(best_pos[0], best_pos[2], c='red', s=300, marker='*', zorder=10,
             label=f'Best grasp')

cbar3R = fig3.colorbar(sc3R, ax=ax3R, shrink=0.8)
cbar3R.set_label('Confidence', color='black', fontsize=10)
cbar3R.ax.yaxis.set_tick_params(color='black')
plt.setp(plt.getp(cbar3R.ax.axes, 'yticklabels'), color='black')

ax3R.set_xlabel('X (m)', color='black', fontsize=11)
ax3R.set_ylabel('Z (m)', color='black', fontsize=11)
ax3R.set_title('(b) Side projection (XZ)', loc='left',
               color='black', fontsize=12)
ax3R.tick_params(colors='black')
ax3R.spines[:].set_color('#777777')
ax3R.legend(facecolor='white', labelcolor='black', edgecolor='#777777',
            framealpha=1.0, fontsize=9)

fig3.suptitle(
    'Contact-GraspNet proposal geometry on the cylinder',
    color='black', fontsize=13, y=1.01,
)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, 'cgn_zoomed_2d_projections.png')
plt.savefig(out3, dpi=160, bbox_inches='tight', facecolor='white')
plt.close(fig3)
print(f'Saved: {out3}')

print('\nAll annotated figures saved to results/figures/')
