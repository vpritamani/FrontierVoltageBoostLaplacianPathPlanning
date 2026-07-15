"""
Frontier RRT–Laplace hybridization (random sampling on the frontier).

An incremental predecessor of Frontier Voltage Boost Laplace. Like the random
sampling variant it grows an RRT-style tree of zero-potential nodes on a plain
harmonic field and **has no voltage boost**, but instead of sampling anywhere in
the diffused region it samples on the *frontier* between the solved region and
free space:

  * The potential is harmonic — obstacles (and the map border) pinned to 1, the
    goal pinned to 0, free space relaxed by 4-neighbour Laplace averaging. The
    goal **and every tree cell** are pinned to 0 each iteration, so the
    low-potential basin widens as the tree grows. Gradient descent uses a plain
    central-difference gradient (no ``_fv`` wavefront cap).

  * A cell is *solved* once ``phi <= 1 - epsilon``. The frontier is the set of
    solved cells adjacent to still-free cells — the boundary produced by a
    simple segmentation of the solved-vs-free masks (done here with NumPy
    neighbour shifts rather than an external edge detector).

Each outer iteration (no warm-up):

  1. Run ``n_l`` Laplace iterations.
  2. Sample up to ``n_r`` points on the frontier and gradient-descend from each;
     a descent that reaches the tree adds its whole branch (nodes pinned to 0,
     parent pointers toward the goal) via an ``rtree`` spatial index.
  3. Once the start is solved, descend from the start into the tree and return
     the path start → … → goal.

``use_gpu`` runs the Laplace averaging as a ``conv2d`` on a PyTorch device
(CUDA when available), matching FVB. The returned path is a list of continuous
``(x, y)`` coordinates from start to goal.
"""

import math
import random

import numpy as np
from rtree import index

from Algorithms import Algorithm2D


class RRTLaplaceFrontier(Algorithm2D):

    def __init__(self, map, n_l: int = 20, n_r: int = 5, epsilon: float = 0.5,
                 step_size: float = 1.0, bilinear_interpolation: bool = True,
                 max_outer_iters: int = 1000, seed: int = -1,
                 use_gpu: bool = False):
        super().__init__(map)
        self.n_l = n_l                              # Laplace iters per outer step
        self.n_r = n_r                              # frontier seeds per outer step
        self.epsilon = epsilon                      # solved when phi <= 1 - epsilon
        self.step_size = step_size
        self.bilinear_interpolation = bilinear_interpolation
        self.max_outer_iters = max_outer_iters
        self.seed = seed                            # -1 = fresh randomness
        self.use_gpu = use_gpu
        self.connect_radius = max(1.5, float(step_size))
        # A seed within this of the tree counts as already on it (skipped for
        # growth); kept well below connect_radius so a frontier seed one cell
        # outside the tree still carves a genuine short branch.
        self.skip_radius = 0.5
        self._path = None
        self._phi = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self, visualize=False):
        grid = self.map.grid
        self._H, self._W = grid.shape
        sx, sy = int(self.map.start[0]), int(self.map.start[1])
        ex, ey = int(self.map.end[0]), int(self.map.end[1])
        self._start = (sx, sy)
        self._goal = (ex, ey)

        self._obstacle = self._obstacle_mask(grid)
        if self._obstacle[sy, sx] or self._obstacle[ey, ex]:
            self._path = None
            return None

        # Harmonic potential: obstacles/border = 1, goal = 0, free relaxes.
        phi = np.ones(grid.shape, dtype=np.float64)
        phi[self._obstacle] = 1.0
        phi[ey, ex] = 0.0

        # Tree rooted at the goal (node 0). Nodes store continuous coords; each
        # non-root node's parent points one step toward the goal.
        self._nodes = [(float(ex), float(ey))]
        self._parent = [-1]
        self._tree_mask = np.zeros(grid.shape, dtype=bool)
        self._tree_mask[ey, ex] = True
        self._tree_index = index.Index()
        self._tree_index.insert(0, (float(ex), float(ey), float(ex), float(ey)))

        self._max_descent_steps = max(1000, 4 * (self._H + self._W))
        rng = random.Random(self.seed) if self.seed >= 0 else random.Random()

        solved_thresh = 1.0 - self.epsilon

        for _ in range(self.max_outer_iters):
            phi = self._laplace(phi, self.n_l)
            self._phi = phi

            # Grow the tree from frontier seeds.
            for sxp, syp in self._sample_seeds(phi, rng, solved_thresh):
                res = self._descend(phi, sxp, syp)
                if res is not None:
                    self._add_branch(*res)

            # Termination: is the start solved? If so, descend from the start
            # into the tree and return the path.
            if phi[sy, sx] <= solved_thresh:
                res = self._descend(phi, float(sx), float(sy), allow_immediate=True)
                if res is not None:
                    self._path = self._reconstruct(*res)
                    if visualize:
                        self._visualize()
                    return self._path

        self._path = None
        return None

    def result(self, visualize=False):
        if visualize:
            self._visualize()
        return self._path

    # ------------------------------------------------------------------
    # Seed sampling — points on the solved/free frontier
    # ------------------------------------------------------------------

    def _sample_seeds(self, phi, rng, solved_thresh):
        solved = (phi <= solved_thresh) & (~self._obstacle)
        free = (~solved) & (~self._obstacle)

        # Frontier = *free* cells with at least one solved 4-neighbour — the
        # boundary from segmenting the solved-vs-free masks, taken on the free
        # side (neighbour shifts, the dependency-free equivalent of edge
        # detection). Seeding just outside the solved region keeps every seed a
        # step beyond the tree, so each descent carves a genuinely new branch
        # inward to the tree (a seed *inside* the tree would connect trivially).
        frontier = np.zeros_like(free)
        frontier[:-1, :] |= free[:-1, :] & solved[1:, :]   # solved below
        frontier[1:, :] |= free[1:, :] & solved[:-1, :]    # solved above
        frontier[:, :-1] |= free[:, :-1] & solved[:, 1:]   # solved right
        frontier[:, 1:] |= free[:, 1:] & solved[:, :-1]    # solved left

        pts = np.argwhere(frontier)
        seeds = []
        for _ in range(self.n_r):
            if len(pts):
                yy, xx = pts[rng.randrange(len(pts))]
                seeds.append((float(xx), float(yy)))
        return seeds

    # ------------------------------------------------------------------
    # Potential field — masks + Laplace relaxation (CPU / GPU)
    # ------------------------------------------------------------------

    @staticmethod
    def _obstacle_mask(grid):
        obstacle = grid == 1
        obstacle[0, :] = True
        obstacle[-1, :] = True
        obstacle[:, 0] = True
        obstacle[:, -1] = True
        return obstacle

    def _laplace(self, phi, n):
        if n <= 0:
            return phi
        if self.use_gpu:
            return self._laplace_gpu(phi, n)
        return self._laplace_cpu(phi, n)

    def _laplace_cpu(self, phi, n):
        """4-neighbour Laplace averaging, re-pinning boundary conditions each
        iteration: obstacles → 1, goal and every tree cell → 0."""
        ex, ey = self._goal
        obstacle, tree = self._obstacle, self._tree_mask
        for _ in range(n):
            padded = np.pad(phi, 1, mode='edge')
            phi = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                   padded[1:-1, :-2] + padded[1:-1, 2:]) * 0.25
            phi[obstacle] = 1.0
            phi[tree] = 0.0
            phi[ey, ex] = 0.0
        return phi

    def _laplace_gpu(self, phi_np, n):
        """Same relaxation with averaging expressed as a conv2d (FVB's kernel)."""
        import torch
        import torch.nn.functional as tF

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        kernel = torch.tensor([[0.0, 0.25, 0.0],
                               [0.25, 0.0, 0.25],
                               [0.0, 0.25, 0.0]],
                              dtype=torch.double, device=device).view(1, 1, 3, 3)
        phi = torch.from_numpy(phi_np).to(device)
        obstacle = torch.from_numpy(self._obstacle).to(device)
        tree = torch.from_numpy(self._tree_mask).to(device)
        ones = torch.ones_like(phi)
        zeros = torch.zeros_like(phi)
        ex, ey = self._goal
        for _ in range(n):
            padded = tF.pad(phi.view(1, 1, *phi.shape), (1, 1, 1, 1), mode='replicate')
            new = tF.conv2d(padded, kernel).view(*phi.shape)
            new = torch.where(obstacle, ones, new)
            new = torch.where(tree, zeros, new)
            new[ey, ex] = 0.0
            phi = new
        return phi.cpu().numpy()

    # ------------------------------------------------------------------
    # Gradient descent + tree growth
    # ------------------------------------------------------------------

    def _query_tree(self, x, y, radius):
        """Return the id of the nearest tree node within ``radius``, else None."""
        best, best_d2 = None, radius * radius
        for nid in self._tree_index.intersection((x - radius, y - radius,
                                                  x + radius, y + radius)):
            nx, ny = self._nodes[nid]
            d2 = (nx - x) ** 2 + (ny - y) ** 2
            if d2 <= best_d2:
                best_d2, best = d2, nid
        return best

    def _descend(self, phi, sx, sy, allow_immediate=False):
        """Constant-speed steepest descent from (sx, sy). Returns
        ``(points, connected_id)`` when it reaches the tree, else None.
        ``points`` are continuous (x, y) from the seed to the connection."""
        posx, posy = float(sx), float(sy)
        points = [(posx, posy)]

        # The start's terminal descent connects immediately if already within
        # connect_radius; a growth seed is only skipped when it essentially sits
        # on a node (skip_radius), so a near-tree frontier seed still descends.
        init_r = self.connect_radius if allow_immediate else self.skip_radius
        hit = self._query_tree(posx, posy, init_r)
        if hit is not None:
            return (points, hit) if allow_immediate else None

        for _ in range(self._max_descent_steps):
            gx, gy = self._grad(phi, posx, posy)
            mag = math.hypot(gx, gy)
            if mag == 0.0:
                return None
            posx -= self.step_size * gx / mag
            posy -= self.step_size * gy / mag
            if not (1 <= posy < self._H - 1 and 1 <= posx < self._W - 1):
                return None
            points.append((posx, posy))
            hit = self._query_tree(posx, posy, self.connect_radius)
            if hit is not None:
                return points, hit
        return None

    def _add_branch(self, points, connected_id):
        """Append a descent branch to the tree with parent pointers running from
        the seed toward the tree (last node → the node it connected to)."""
        base = len(self._nodes)
        n = len(points)
        for i, (x, y) in enumerate(points):
            nid = base + i
            parent = (base + i + 1) if i < n - 1 else connected_id
            self._nodes.append((x, y))
            self._parent.append(parent)
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < self._H and 0 <= xi < self._W:
                self._tree_mask[yi, xi] = True
            self._tree_index.insert(nid, (x, y, x, y))

    def _reconstruct(self, points, connected_id):
        """Full start→goal path: the start's descent points, then the parent
        chain from the connected node down to the goal (all continuous)."""
        path = list(points)
        nid = connected_id
        guard, limit = 0, len(self._nodes) + 5
        while nid != -1 and guard < limit:
            path.append(self._nodes[nid])
            nid = self._parent[nid]
            guard += 1
        return path

    # ------------------------------------------------------------------
    # Gradient sampling (plain central difference — no ``_fv`` cap)
    # ------------------------------------------------------------------

    def _grad(self, phi, x, y):
        if self.bilinear_interpolation:
            gx = self._bilinear(phi, x + 1, y) - self._bilinear(phi, x - 1, y)
            gy = self._bilinear(phi, x, y + 1) - self._bilinear(phi, x, y - 1)
        else:
            xi, yi = math.floor(x), math.floor(y)
            gx = phi[yi, xi + 1] - phi[yi, xi - 1]
            gy = phi[yi + 1, xi] - phi[yi - 1, xi]
        return gx, gy

    @staticmethod
    def _bilinear(phi, x, y):
        """Sample phi at continuous (x, y); grid is indexed phi[y, x]."""
        y0, x0 = math.floor(y), math.floor(x)
        y1, x1 = y0 + 1, x0 + 1
        y0 = max(0, min(y0, phi.shape[0] - 1))
        y1 = max(0, min(y1, phi.shape[0] - 1))
        x0 = max(0, min(x0, phi.shape[1] - 1))
        x1 = max(0, min(x1, phi.shape[1] - 1))
        dy = y - math.floor(y)
        dx = x - math.floor(x)
        return ((1 - dy) * (1 - dx) * phi[y0, x0]
                + (1 - dy) * dx * phi[y0, x1]
                + dy * (1 - dx) * phi[y1, x0]
                + dy * dx * phi[y1, x1])

    # ------------------------------------------------------------------
    # Visualization (2D)
    # ------------------------------------------------------------------

    def _visualize(self):
        from PIL import Image
        grey = self.map.to_image()
        rgb = np.stack([grey, grey, grey], axis=2).copy()
        if self._path:
            for x, y in self._path:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < rgb.shape[0] and 0 <= xi < rgb.shape[1]:
                    rgb[yi, xi] = [128, 128, 255]
        sx, sy = self.map.start
        ex, ey = self.map.end
        rgb[sy, sx] = [0, 255, 0]
        rgb[ey, ex] = [255, 0, 0]
        Image.fromarray(rgb, mode='RGB').save('rrt_laplace_frontier_result.png')
