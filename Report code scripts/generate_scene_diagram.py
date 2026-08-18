import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import os

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw object (a cube) at the origin
r = [-0.5, 0.5]
X, Y = np.meshgrid(r, r)
ax.plot_surface(X, Y, np.full_like(X, 0.5), alpha=0.5, color='orange')
ax.plot_surface(X, Y, np.full_like(X, -0.5), alpha=0.5, color='orange')
ax.plot_surface(X, np.full_like(X, -0.5), Y, alpha=0.5, color='orange')
ax.plot_surface(X, np.full_like(X, 0.5), Y, alpha=0.5, color='orange')
ax.plot_surface(np.full_like(X, -0.5), X, Y, alpha=0.5, color='orange')
ax.plot_surface(np.full_like(X, 0.5), X, Y, alpha=0.5, color='orange')

# Draw axes
arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
a = Arrow3D([0, 2], [0, 0], [0, 0], **arrow_prop_dict)
ax.add_artist(a)
a = Arrow3D([0, 0], [0, 2], [0, 0], **arrow_prop_dict)
ax.add_artist(a)
a = Arrow3D([0, 0], [0, 0], [0, 2], **arrow_prop_dict)
ax.add_artist(a)

ax.text(2.1, 0, 0, 'X')
ax.text(0, 2.1, 0, 'Y')
ax.text(0, 0, 2.1, 'Z')

# Draw camera and view vector
r_cam = 4
theta = np.deg2rad(45)
phi = np.deg2rad(30)
xc = r_cam * np.cos(phi) * np.cos(theta)
yc = r_cam * np.cos(phi) * np.sin(theta)
zc = r_cam * np.sin(phi)

# View vector
ax.plot([0, xc], [0, yc], [0, zc], 'k--', alpha=0.5)

# Camera box
ax.scatter([xc], [yc], [zc], c='blue', marker='s', s=100)
ax.text(xc, yc, zc+0.3, 'Camera', color='blue', fontsize=12)

# Draw arc for theta (azimuth)
theta_vals = np.linspace(0, theta, 50)
x_arc_theta = 1.5 * np.cos(theta_vals)
y_arc_theta = 1.5 * np.sin(theta_vals)
z_arc_theta = np.zeros_like(x_arc_theta)
ax.plot(x_arc_theta, y_arc_theta, z_arc_theta, 'r-', linewidth=2)
ax.text(1.6*np.cos(theta/2), 1.6*np.sin(theta/2), 0, r'$\theta$ (Azimuth)', color='red', fontsize=16)

# Draw arc for phi (elevation)
phi_vals = np.linspace(0, phi, 50)
x_arc_phi = 1.5 * np.cos(phi_vals) * np.cos(theta)
y_arc_phi = 1.5 * np.cos(phi_vals) * np.sin(theta)
z_arc_phi = 1.5 * np.sin(phi_vals)
ax.plot(x_arc_phi, y_arc_phi, z_arc_phi, 'g-', linewidth=2)
ax.plot([0, xc], [0, yc], [0, 0], 'k:', alpha=0.5) # Projection on XY plane
ax.text(1.6*np.cos(phi/2)*np.cos(theta), 1.6*np.cos(phi/2)*np.sin(theta), 1.6*np.sin(phi/2), r'$\phi$ (Elevation)', color='green', fontsize=16)

# Labels for rho and sigma_d
ax.text(0, -2.5, 2.5, r'$\sigma_d$ (Depth Noise)', color='purple', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
ax.text(0, -2.5, 2.0, r'$\rho$ (Point Cloud Sparsity)', color='brown', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

ax.set_xlim([-1, 4])
ax.set_ylim([-1, 4])
ax.set_zlim([-1, 4])
ax.set_axis_off()

os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/scene_diagram.png', bbox_inches='tight', dpi=300)
