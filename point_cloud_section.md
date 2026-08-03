# Point Cloud — Verbal Defence Guide

Use this when a professor, examiner, or peer asks "what is a point cloud?"
Answer at whichever depth the context demands.

---

## One sentence (if interrupted mid-presentation)

> "A point cloud is a finite, unordered set of 3D coordinates — one per surface point
> the depth sensor could measure — encoding the visible geometry of the scene."

---

## Two minutes (if asked to explain properly)

**What it is, mathematically:**

A point cloud $\mathcal{P} = \{\mathbf{p}_1, \ldots, \mathbf{p}_N\}$ where each
$\mathbf{p}_i = (X_i, Y_i, Z_i)^\top \in \mathbb{R}^3$ is the 3D location of one
surface sample. That's the entire data structure. No topology, no connectivity,
no ordering. Just N three-dimensional vectors.

**How it arises in this project:**

The MuJoCo camera renders a depth image — a 2D array where each pixel stores the
distance to the nearest surface. Back-projection inverts the pinhole camera model:

$$X = \frac{(u - c_x) \cdot d}{f_x}, \quad Y = \frac{(v - c_y) \cdot d}{f_y}, \quad Z = d$$

Every valid pixel becomes one point. The full set of those points is the point cloud.

**Why it's partial:**

The camera can only see surfaces it has line-of-sight to. Occluded surfaces, and
surfaces beyond the sensor range, produce no depth measurement and therefore no point.
This is not a bug — it's the physically accurate model of what a real depth sensor
(RealSense, Kinect, Azure Kinect) actually returns.

**What the causal variables do to it:**

| Variable | Effect on the point cloud |
|---|---|
| $\sigma_d$ (depth noise) | Perturbs each $Z_i$ value → distorts the 3D geometry of every point |
| $\rho$ (sparsity) | Randomly removes $(1-\rho)$ fraction of points → reduces $N$ |
| $\phi$ (elevation) | Changes which surfaces are visible → changes which points can exist |
| $\theta$ (azimuth) | Changes viewing angle → changes which surfaces are visible |

---

## Five minutes (if they push on "why not a depth image, voxel grid, or mesh?")

**CGN requires a point cloud — it's not a choice.** Contact-GraspNet's architecture
(PointNet++ encoder + per-point contact-prediction heads) takes in a set of 3D
points and outputs, per point, a probability that that surface location is a valid
gripper contact point. You cannot feed it a 2D image or a mesh without rewriting the
entire architecture.

**Partial observability is the *right* distribution.** CGN was trained on partial
point clouds — single-viewpoint depth renders of ShapeNet objects — exactly as
produced by back-projection. Feeding it a complete CAD model would be out of its
training distribution and would misrepresent the real sensing situation.

**Causal perturbations are cleanly algebraic on the point set.** Depth noise perturbs
coordinates. Sparsity removes points. Viewpoint changes which points exist. These
three operations are orthogonal and act on well-defined mathematical objects.
On a depth image, the same operations have entangled effects (noise at one pixel
shifts the apparent 3D position of the *next* pixel's neighbourhood when viewed in 3D).
On a mesh, removing points destroys topology. The point cloud is the only
representation where all four causal variables have clean, non-interacting semantics.

---

## What NOT to say again

❌ "A point cloud is the mathematical equivalent of an RGB image but with points instead."

This is wrong because:
- An RGB image is a *grid* (pixels have fixed neighbours). A point cloud is an *unordered set*.
- An RGB image encodes *appearance* (colour). A point cloud encodes *geometry* (position in 3D space).
- The correct analogy (if forced to give one): "A depth image is to a point cloud as a
  2D array of distances is to the 3D surface those distances encode."

---

## What was added to the thesis

A new section **"The Point Cloud as a Geometric Representation"** was inserted into
`msc_report.tex` (around line 635, before the CGN visualisation section) containing:

1. **Formal Definition** — set-theoretic definition with a numbered Definition environment
2. **Generation via Pinhole Back-Projection** — full derivation from depth buffer to world-frame point set
3. **Why the Point Cloud Is the Correct Representation** — three arguments:
   - CGN's architecture mandates it
   - Partial observability matches CGN's training distribution
   - Causal perturbations are cleanly separable on the point cloud
4. **Formal Definition of the Downsampled Point Cloud** — precise combinatorial definition
   of random subsampling with the `rng.choice` implementation note

The `qi2017pointnetpp` (PointNet++) citation was also added to the bibliography.

The old one-paragraph back-projection section and the duplicate focal-length derivation
in the Coordinate Frame Transformation section were replaced with cross-references to
the new section.
