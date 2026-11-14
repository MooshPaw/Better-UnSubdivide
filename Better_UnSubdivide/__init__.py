# SPDX-License-Identifier: GPL-3.0-or-later
"""
Un-Subdivide (Seam Aware)

A destructive, button-based version of the Decimate modifier's "Un-Subdivide"
mode. Because it edits the mesh data directly (instead of being a modifier),
it works on meshes that have shape keys: every shape key is carried over.

The core is a Python port of Blender's own un-subdivide
(source/blender/bmesh/tools/bmesh_decimate_unsubdivide.cc):

* Each iteration marks dissolvable verts (3/4 manifold edges, border verts
  with 2 border edges + 1 manifold edge, or 2-edge chains), then flood-fills
  outward from a seed, alternating "keep" / "collapse" per ring.
* Collapse verts are dissolved "as a fan": every n-gon corner around the vert
  is clipped into a triangle ear, then the vert is dissolved, so the
  surrounding faces merge into one diamond. One iteration halves the density;
  2 iterations undo one subdivision level.

Additions on top of Blender's algorithm:

* Seam awareness: verts touching a protected edge (seam, UV split, sharp,
  material border) are never fan-dissolved, so protected loops are never cut
  or zig-zagged. Verts lying *on* a seam (exactly 2 protected edges) still
  take part in the pattern and are removed as a chain (merging the two seam
  edges into one) once the faces on both sides have been reduced, so the seam
  stays continuous but gets the same clean, halved resolution.
* Mirror: the pattern is grown from mirrored seeds and every tag is copied to
  the mirrored vertex, so both halves are decimated identically.
"""

bl_info = {
    "name": "Better Un-Subdivide",
    "author": "MooshPaw",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > Tool",
    "description": "Destructive Un-Subdivide that keeps shape keys, respects UV seams "
                   "and can stay mirrored",
    "category": "Mesh",
}

import bmesh
import bpy
import numpy as np
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty)
from mathutils import kdtree

_ORIG_LAYER = "_unsubdiv_orig_index"
_UV_EPS = 1e-6

INIT, IGNORE, COLLAPSE = 0, 1, -1
_DEBUG = None  # test hook
FAN, BOUNDARY_FAN, CHAIN = 1, 2, 3


class Options:
    def __init__(self, seams=True, uv_islands=False, sharp=False,
                 materials=True, selected_only=False, flip=False,
                 mirror=True, mirror_axis=0, mirror_tol=1e-4, cleanup=True,
                 join_face_angle=1.0472, join_shape_angle=1.0472,
                 join_topology=1.0):
        self.seams = seams
        self.uv_islands = uv_islands
        self.sharp = sharp
        self.materials = materials
        self.selected_only = selected_only
        self.flip = flip
        self.mirror = mirror
        self.mirror_axis = mirror_axis
        self.mirror_tol = mirror_tol
        self.cleanup = cleanup
        self.join_face_angle = join_face_angle
        self.join_shape_angle = join_shape_angle
        self.join_topology = join_topology

    @property
    def any_protection(self):
        return self.seams or self.uv_islands or self.sharp or self.materials


# --------------------------------------------------------------------------
# Protection (seams etc.)
# --------------------------------------------------------------------------

def _make_protect_test(bm, opts):
    uv_layers = list(bm.loops.layers.uv) if opts.uv_islands else []
    if not opts.any_protection:
        return lambda e: False

    def is_protected(e):
        if opts.seams and e.seam:
            return True
        if opts.sharp and not e.smooth:
            return True
        loops = e.link_loops
        if len(loops) != 2:
            return False
        l1, l2 = loops
        if opts.materials and l1.face.material_index != l2.face.material_index:
            return True
        for uv in uv_layers:
            a = {l1.vert: l1[uv].uv, l1.link_loop_next.vert: l1.link_loop_next[uv].uv}
            b = {l2.vert: l2[uv].uv, l2.link_loop_next.vert: l2.link_loop_next[uv].uv}
            for v in e.verts:
                if (a[v] - b[v]).length_squared > _UV_EPS:
                    return True
        return False

    return is_protected


def _seam_passthrough(v, is_protected):
    """Vert lying on a protected line: exactly 2 protected manifold edges."""
    n = 0
    for e in v.link_edges:
        if is_protected(e):
            if not e.is_manifold:
                return False
            n += 1
    return n == 2


# --------------------------------------------------------------------------
# Port of bm_vert_dissolve_fan_or_chain_test / bm_vert_dissolve_fan_or_chain
# --------------------------------------------------------------------------

def _dissolve_test(v, is_protected, passthrough):
    """Return the dissolve method for `v` or None (mirrors Blender's test)."""
    n = nb = nm = nw = relaxed = 0
    for e in v.link_edges:
        if is_protected(e):
            return None
        nf = len(e.link_faces)
        if nf == 1:
            nb += 1
        elif nf == 2:
            nm += 1
            # Seam-aware extension: an extra edge to a seam vert whose two
            # faces are triangles (ears left from the previous step) doesn't
            # count towards the valence limit, so the seam vert can be freed.
            if passthrough is not None:
                o = e.other_vert(v)
                if passthrough(o) and all(len(f.verts) == 3 for f in e.link_faces):
                    relaxed += 1
        elif nf == 0:
            nw += 1
        else:
            return None
        n += 1
        if n > 4 + relaxed + 2:
            return None
    eff = n - relaxed
    if nb == 0 and nm == n and n >= 3 and eff <= 4:
        return FAN
    if n == 3 and nb == 2 and nm == 1:
        return BOUNDARY_FAN
    if n == 2 and (nw == 2 or nm == 2):
        return CHAIN
    return None


def _edge_between(a, b):
    for e in a.link_edges:
        if e.other_vert(a) is b:
            return e
    return None


def _face_exists(verts, exclude=()):
    vs = set(verts)
    n = len(vs)
    for f in verts[0].link_faces:
        if f in exclude or len(f.verts) != n:
            continue
        if all(x in vs for x in f.verts):
            return True
    return False


def _fan_new_faces(v, skip_faces=()):
    """Vertex sets of the faces a fan dissolve of `v` would leave behind
    (the merged face, plus the trimmed n-gons not in `skip_faces`)."""
    out = []
    ring = set()
    for l in v.link_loops:
        f = l.face
        if len(f.verts) > 3 and f not in skip_faces:
            out.append(frozenset(x for x in f.verts if x is not v))
        ring.add(l.link_loop_prev.vert)
        ring.add(l.link_loop_next.vert)
    out.append(frozenset(ring))
    return out


def _fan_makes_double(v, predicted=(), shared=()):
    """Port of Blender's duplicate-face guards for a fan dissolve.
    `predicted`: vertex sets of faces another pending dissolve will leave;
    `shared`: faces shared with that other vert (handled sequentially)."""
    loops = list(v.link_loops)
    vfaces = set(l.face for l in loops)
    ring = set()
    collapsed = []
    for l in loops:
        f = l.face
        prev_v, next_v = l.link_loop_prev.vert, l.link_loop_next.vert
        if len(f.verts) > 3:
            # BM_face_split_check_double_face: the ear (prev, v, next) or the
            # remainder of the face must not already exist as a face.
            if _face_exists([prev_v, v, next_v], exclude=(f,)):
                return True
            rest = [x for x in f.verts if x is not v]
            if _face_exists(rest, exclude=(f,)):
                return True
            if f not in shared and frozenset(rest) in predicted:
                return True
            # Remainders of two faces around v collapsing to the same face.
            key = frozenset(rest)
            if key in collapsed:
                return True
            collapsed.append(key)
        ring.add(prev_v)
        ring.add(next_v)
    # The merged face around v would duplicate an existing face.
    r = list(ring)
    if len(r) >= 3 and (_face_exists(r, exclude=vfaces) or frozenset(r) in predicted):
        return True
    return False


def _dissolve_vert(bm, v, method):
    if method == CHAIN:
        e = v.link_edges[0]
        bmesh.utils.vert_collapse_edge(v, e)
        return True
    if _fan_makes_double(v):
        return False
    for l in list(v.link_loops):
        f = l.face
        if len(f.verts) > 3:
            e_a, e_b = l.edge, l.link_loop_prev.edge
            seam = e_a.seam or e_b.seam
            smooth = e_a.smooth and e_b.smooth
            _f, l_new = bmesh.utils.face_split(f, l.link_loop_prev.vert,
                                               l.link_loop_next.vert)
            if l_new is not None:
                l_new.edge.seam = seam
                l_new.edge.smooth = smooth
    return bmesh.utils.vert_dissolve(v)


# --------------------------------------------------------------------------
# Mirror map
# --------------------------------------------------------------------------

def _mirror_map(bm, opts):
    if not opts.mirror:
        return None
    axis = opts.mirror_axis
    verts = bm.verts
    tree = kdtree.KDTree(len(verts))
    for v in verts:
        tree.insert(v.co, v.index)
    tree.balance()
    size = max((max(v.co[i] for v in verts) - min(v.co[i] for v in verts)
                for i in range(3)), default=1.0) if len(verts) else 1.0
    tol = max(opts.mirror_tol, size * 1e-5)
    mirror = [None] * len(verts)
    for v in verts:
        co = v.co.copy()
        co[axis] = -co[axis]
        _co, idx, dist = tree.find(co)
        if idx is not None and dist <= tol:
            mirror[v.index] = verts[idx]
    return mirror


# --------------------------------------------------------------------------
# One iteration (port of BM_mesh_decimate_unsubdivide_ex inner loop)
# --------------------------------------------------------------------------

def _tag_and_dissolve(bm, opts, flips=None, comp_layer=None):
    """One Blender-style un-subdivide step.

    flips: optional dict {region id: bool} choosing the parity per region.
    comp_layer: optional int vert layer receiving each vert's region id.
    Returns (removed vert count, number of regions).
    """
    bm.verts.index_update()
    bm.verts.ensure_lookup_table()
    is_protected = _make_protect_test(bm, opts)
    use_pt = opts.any_protection
    pt_cache = {}

    def passthrough(v):
        r = pt_cache.get(v)
        if r is None:
            r = pt_cache[v] = _seam_passthrough(v, is_protected)
        return r

    touch_cache = {}

    def touches_protected(v):
        # Any vert on a protected line (incl. junctions and line ends): it is
        # never fan-dissolved, so triangle ears pointing at it may be merged.
        r = touch_cache.get(v)
        if r is None:
            r = touch_cache[v] = any(is_protected(e) for e in v.link_edges)
        return r

    pt_fn = touches_protected if use_pt else None
    mirror = _mirror_map(bm, opts)

    nv = len(bm.verts)
    tag = [IGNORE] * nv
    comp = [-1] * nv
    deferred = [False] * nv
    eligible = [False] * nv
    for v in bm.verts:
        if opts.selected_only and not v.select:
            continue
        if _dissolve_test(v, is_protected, pt_fn) is not None:
            tag[v.index] = INIT
            eligible[v.index] = True
        elif use_pt and passthrough(v) and len(v.link_edges) <= 4:
            # Seam vert: takes part in the pattern, only removed as a chain.
            tag[v.index] = INIT
            deferred[v.index] = True

    def expand(starts):
        """Tag untagged eligible neighbours with the opposite of their
        source's tag (identical to Blender's ring tagging, but also works
        when a ring starts from many mixed seeds)."""
        out = []
        for v in starts:
            t = -tag[v.index]
            for e in v.link_edges:
                o = e.other_vert(v)
                oi = o.index
                if tag[oi] == INIT:
                    tag[oi] = t
                    out.append(o)
                    if mirror is not None:
                        m = mirror[oi]
                        if m is not None and tag[m.index] == INIT:
                            tag[m.index] = t
                            out.append(m)
        return out

    def seed(v, t):
        tag[v.index] = t
        out = [v]
        if mirror is not None:
            m = mirror[v.index]
            if m is not None and tag[m.index] == INIT:
                tag[m.index] = t
                out.append(m)
        return out

    # Islands (connected mesh pieces) and a reference two-colouring of each.
    # The colouring decides which parity every flood fill below starts with,
    # so separate fills (split by poles or seams, as in Blender) still agree
    # with each other. On subdivided topology this is globally consistent.
    island = [-1] * nv
    ref = [0] * nv
    nisland = 0
    for v0 in bm.verts:
        if island[v0.index] != -1:
            continue
        cid = nisland
        nisland += 1
        island[v0.index] = cid
        ring = [v0]
        while ring:
            nxt = []
            for v in ring:
                for e in v.link_edges:
                    o = e.other_vert(v)
                    if island[o.index] == -1:
                        island[o.index] = cid
                        nxt.append(o)
            ring = nxt
    for v0 in bm.verts:
        if ref[v0.index]:
            continue
        ref[v0.index] = 1
        group = [v0]
        if mirror is not None:
            m = mirror[v0.index]
            if m is not None and not ref[m.index]:
                ref[m.index] = 1
                group.append(m)
        ring = group
        while ring:
            nxt = []
            for v in ring:
                c = -ref[v.index]
                for e in v.link_edges:
                    if use_pt and is_protected(e):
                        continue
                    o = e.other_vert(v)
                    if not ref[o.index]:
                        ref[o.index] = c
                        nxt.append(o)
                        if mirror is not None:
                            m = mirror[o.index]
                            if m is not None and not ref[m.index]:
                                ref[m.index] = c
                                nxt.append(m)
            ring = nxt
    for i in range(nv):
        comp[i] = island[i]

    # Which colour to keep, per island. On subdivided topology the verts
    # removed first (edge midpoints) are always regular: 4 edges inside, 3 on
    # a border, never touching a seam junction/end. Irregular verts (poles,
    # corners, seam junctions, ...) therefore mark the colour to keep.
    irr_pos = [0] * nisland
    irr_neg = [0] * nisland
    for v in bm.verts:
        n = len(v.link_edges)
        border = False
        bad = False
        nprot = 0
        for e in v.link_edges:
            nf = len(e.link_faces)
            if nf == 1:
                border = True
            elif nf != 2:
                bad = True
            if use_pt and is_protected(e):
                nprot += 1
        irregular = bad or n != (3 if border else 4) or nprot not in (0, 2)
        if irregular:
            if ref[v.index] == 1:
                irr_pos[island[v.index]] += 1
            else:
                irr_neg[island[v.index]] += 1
    # Only trust this when it's lopsided (i.e. one colour is all regular);
    # otherwise keep Blender's default and let trial runs decide.
    keep_color = [1] * nisland
    decided = [False] * nisland
    for i in range(nisland):
        a, b = irr_pos[i], irr_neg[i]
        if max(a, b) >= 2 and min(a, b) * 10 <= max(a, b):
            keep_color[i] = 1 if a > b else -1
            decided[i] = True

    if use_pt:
        # Seam verts left with triangle "ears" by the previous step are
        # waiting for their side neighbours to go: grow the pattern from all
        # of them at once (kept), so exactly those neighbours collapse.
        ring = []
        for v in bm.verts:
            if tag[v.index] == INIT and deferred[v.index]:
                for e in v.link_edges:
                    if not is_protected(e) and len(e.link_faces) == 2 and \
                            all(len(f.verts) == 3 for f in e.link_faces):
                        ring += seed(v, IGNORE)
                        break
        while ring:
            ring = expand(ring)

    # Blender's seeding: first untagged vert starts a flood fill with
    # alternating rings; its starting tag comes from the reference colouring.
    for v in bm.verts:
        if tag[v.index] != INIT:
            continue
        isl = island[v.index]
        flip = opts.flip != bool(flips is not None and not decided[isl]
                                 and flips.get(isl))
        keep = (ref[v.index] == keep_color[isl]) != flip
        ring = seed(v, IGNORE if keep else COLLAPSE)
        while ring:
            ring = expand(ring)
    ncomp = nisland

    if mirror is not None:
        # A vert and its mirror twin that are directly connected can't both
        # go, so keep both. (Twins that merely share a face are fine; real
        # conflicts are caught by the paired dissolve below.)
        for v in bm.verts:
            if tag[v.index] == COLLAPSE:
                m = mirror[v.index]
                if m is not None and m is not v and _edge_between(v, m) is not None:
                    tag[v.index] = tag[m.index] = IGNORE

    if _DEBUG is not None:
        _DEBUG.update(tag=list(tag), deferred=list(deferred))

    if comp_layer is not None:
        for v in bm.verts:
            v[comp_layer] = comp[v.index]

    collapse = [v for v in bm.verts if tag[v.index] == COLLAPSE]
    if not collapse:
        return 0, ncomp
    before = len(bm.verts)
    # Every seam vert is a candidate for the chain merge below, whatever its
    # tag: after its side neighbours are dissolved it's left with just the two
    # seam edges. Only verts whose valence actually dropped are merged.
    seam_chain = [(v, len(v.link_edges)) for v in bm.verts if deferred[v.index]]
    fans = [v for v in collapse if not deferred[v.index]]

    def test(v):
        if not v.is_valid:
            return None
        pt_cache.clear()
        touch_cache.clear()
        return _dissolve_test(v, is_protected, pt_fn)

    if mirror is None:
        for v in fans:
            method = test(v)
            if method is not None:
                _dissolve_vert(bm, v, method)
    else:
        # Dissolve mirror twins together (and skip both if either can't go),
        # so the result doesn't depend on the order verts are processed in.
        done = set()
        for v in fans:
            if v in done:
                continue
            m = mirror[v.index]
            if m is not None and not m.is_valid:
                m = None   # twin already went (it was dissolved as a pair)
            if m is not None and m is not v and eligible[m.index] and \
                    tag[m.index] != COLLAPSE:
                # Twin could go but got the other parity: keep this too.
                # (If the twin simply can't go, e.g. the seams aren't
                # symmetric there, the halves can't match anyway.)
                done.add(v)
                continue
            if m is None or m is v or tag[m.index] != COLLAPSE or deferred[m.index]:
                done.add(v)
                method = test(v)
                if method is not None:
                    _dissolve_vert(bm, v, method)
                continue
            done.add(v)
            done.add(m)
            mv, mm = test(v), test(m)
            if mv is None or mm is None:
                continue
            if mv != CHAIN and mm != CHAIN:
                shared = set(v.link_faces) & set(m.link_faces)
                nv_ = set(_fan_new_faces(v, shared))
                nm_ = set(_fan_new_faces(m, shared))
                if _fan_makes_double(v, nm_, shared) or \
                        _fan_makes_double(m, nv_, shared):
                    continue
            elif (mv != CHAIN and _fan_makes_double(v)) or \
                    (mm != CHAIN and _fan_makes_double(m)):
                continue
            if _dissolve_vert(bm, v, mv):
                mm = test(m)
                if mm is not None:
                    _dissolve_vert(bm, m, mm)

    # Seam verts: merge the two seam edges once the side faces are reduced.
    for v, valence in seam_chain:
        if not v.is_valid or len(v.link_edges) != 2 or valence <= 2:
            continue
        e1, e2 = v.link_edges
        if not (e1.is_manifold and e2.is_manifold):
            continue
        if is_protected(e1) and is_protected(e2) and \
                e1.seam == e2.seam and e1.smooth == e2.smooth:
            bmesh.utils.vert_collapse_edge(v, e1)

    return before - len(bm.verts), ncomp


def _region_scores(bm, comp_layer, ncomp):
    """Per region: (non-quad faces, remaining verts) - lower is better."""
    bad = [0] * ncomp
    verts = [0] * ncomp
    for v in bm.verts:
        c = v[comp_layer]
        if c >= 0:
            verts[c] += 1
    for f in bm.faces:
        if len(f.verts) == 4:
            continue
        for v in f.verts:
            c = v[comp_layer]
            if c >= 0:
                bad[c] += 1
                break
    return bad, verts


def unsubdivide_step(bm, opts):
    """Run one step. Where the topology doesn't tell which half of the verts
    to remove, try both and keep, per island, the one leaving fewer
    triangles / n-gons. Returns removed vert count."""
    trial = {}
    ncomp = 0
    for flip in (False, True):
        work = bm.copy()
        layer = work.verts.layers.int.new("_unsubdiv_comp")
        _removed, ncomp = _tag_and_dissolve(
            work, opts, flips=(_AllTrue() if flip else None), comp_layer=layer)
        trial[flip] = _region_scores(work, layer, ncomp)
        work.free()
    (bad0, kept0), (bad1, kept1) = trial[False], trial[True]
    flips = {c: (bad1[c], kept1[c]) < (bad0[c], kept0[c]) for c in range(ncomp)}
    removed, _ = _tag_and_dissolve(bm, opts, flips=flips)
    return removed


class _AllTrue(dict):
    def get(self, key, default=None):
        return True


def _cleanup(bm, opts, orig_tris, orig_valence2):
    """Polish pass: remove leftover 2-edge verts and join stray triangle pairs
    into quads, preferring quads that follow the existing edge flow."""
    is_protected = _make_protect_test(bm, opts)
    # 1) Leftover 2-edge verts in the middle of faces (T-junction leftovers).
    lone = [v for v in bm.verts
            if len(v.link_edges) == 2 and v not in orig_valence2
            and (not opts.selected_only or v.select)
            and all(e.is_manifold for e in v.link_edges)
            and is_protected(v.link_edges[0]) == is_protected(v.link_edges[1])]
    for v in lone:
        if v.is_valid and len(v.link_edges) == 2:
            bmesh.utils.vert_collapse_edge(v, v.link_edges[0])
    # 2) Join triangles the decimation created back into quads.
    tris = [f for f in bm.faces if len(f.verts) == 3 and f not in orig_tris
            and (not opts.selected_only or f.select)]
    if tris:
        bmesh.ops.join_triangles(
            bm, faces=tris,
            cmp_seam=opts.seams, cmp_sharp=opts.sharp,
            cmp_uvs=opts.uv_islands, cmp_materials=opts.materials,
            angle_face_threshold=opts.join_face_angle,
            angle_shape_threshold=opts.join_shape_angle,
            topology_influence=opts.join_topology)


def unsubdivide_bmesh(bm, iterations, opts):
    orig_tris = {f for f in bm.faces if len(f.verts) == 3}
    orig_valence2 = {v for v in bm.verts if len(v.link_edges) == 2}
    removed = 0
    for _ in range(iterations):
        r = unsubdivide_step(bm, opts)
        if r == 0:
            break
        removed += r
    if opts.cleanup:
        before = len(bm.verts)
        _cleanup(bm, opts, orig_tris, orig_valence2)
        removed += before - len(bm.verts)
    return removed


def unsubdivide_mesh(me, iterations, opts):
    """Apply to object-mode mesh data, keeping all shape keys.

    Returns (verts_before, verts_after).
    """
    n_before = len(me.vertices)

    keys = me.shape_keys.key_blocks if me.shape_keys else []
    saved = []
    for kb in keys:
        co = np.empty(n_before * 3, dtype=np.float32)
        kb.data.foreach_get("co", co)
        saved.append(co.reshape(-1, 3))

    bm = bmesh.new()
    bm.from_mesh(me)
    layer = bm.verts.layers.int.new(_ORIG_LAYER)
    for i, v in enumerate(bm.verts):
        v[layer] = i

    unsubdivide_bmesh(bm, iterations, opts)

    orig = np.fromiter((v[layer] for v in bm.verts), dtype=np.int64,
                       count=len(bm.verts))
    bm.verts.layers.int.remove(layer)
    bm.to_mesh(me)
    bm.free()

    # Re-write shape keys explicitly from the original data (verts are only
    # ever removed, never moved or created, so this is exact).
    if me.shape_keys:
        n_after = len(me.vertices)
        for kb, co in zip(me.shape_keys.key_blocks, saved):
            if len(kb.data) == n_after:
                kb.data.foreach_set("co", co[orig].ravel())
    me.update()
    return n_before, len(me.vertices)


# --------------------------------------------------------------------------
# Blender UI
# --------------------------------------------------------------------------

_AXES = [('0', "X", "Mirror across the local X axis"),
         ('1', "Y", "Mirror across the local Y axis"),
         ('2', "Z", "Mirror across the local Z axis")]

_PROP_NAMES = ("iterations", "respect_seams", "respect_uv_islands",
               "respect_sharp", "respect_materials", "selected_only",
               "flip_pattern", "use_mirror", "mirror_axis", "use_cleanup",
               "cleanup_angle")


def _opts_from(p):
    return Options(seams=p.respect_seams, uv_islands=p.respect_uv_islands,
                   sharp=p.respect_sharp, materials=p.respect_materials,
                   selected_only=p.selected_only, flip=p.flip_pattern,
                   mirror=p.use_mirror, mirror_axis=int(p.mirror_axis),
                   cleanup=p.use_cleanup, join_face_angle=p.cleanup_angle,
                   join_shape_angle=p.cleanup_angle)


class _CommonProps:
    iterations: IntProperty(
        name="Iterations", default=2, min=1, max=20,
        description="Same as the Decimate modifier: each iteration halves the "
                    "density, 2 iterations undo one subdivision level")
    use_mirror: BoolProperty(
        name="Keep Mirrored", default=True,
        description="Decimate both halves of a symmetric mesh identically")
    mirror_axis: EnumProperty(name="Axis", items=_AXES, default='0')
    use_cleanup: BoolProperty(
        name="Tidy Triangles", default=True,
        description="After decimating, join leftover triangle pairs into quads "
                    "that follow the existing edge flow (never across "
                    "protected edges; triangles that were already in the "
                    "mesh are left alone)")
    cleanup_angle: FloatProperty(
        name="Max Angle", default=1.0472, min=0.0, max=3.14159,
        subtype='ANGLE',
        description="How far from flat/square a joined quad may be. Higher "
                    "joins more triangles but allows more warped quads")
    respect_seams: BoolProperty(
        name="Respect Seams", default=True,
        description="Never cut UV seam loops. Vertices along a seam are still "
                    "reduced, keeping the seam continuous")
    respect_uv_islands: BoolProperty(
        name="Respect UV Splits", default=False,
        description="Also protect edges where the UVs are split but not marked "
                    "as seams")
    respect_sharp: BoolProperty(
        name="Respect Sharp Edges", default=False,
        description="Protect edges marked Sharp")
    respect_materials: BoolProperty(
        name="Respect Material Borders", default=True,
        description="Protect edges between faces with different materials")
    flip_pattern: BoolProperty(
        name="Flip Pattern", default=False,
        description="The add-on picks, per mesh region, whichever half of the "
                    "vertices gives the cleaner result. This inverts that "
                    "choice. Try it if the result doesn't line up with your "
                    "edge loops")
    selected_only: BoolProperty(
        name="Selected Only", default=False,
        description="Only dissolve selected vertices (Edit Mode selection)")


class UNSUBDIV_PG_settings(_CommonProps, bpy.types.PropertyGroup):
    pass


def _target_meshes(context):
    objs = [o for o in context.selected_objects if o.type == 'MESH']
    if context.active_object and context.active_object.type == 'MESH' \
            and context.active_object not in objs:
        objs.append(context.active_object)
    seen, meshes = set(), []
    for o in objs:
        if o.data.name_full not in seen:
            seen.add(o.data.name_full)
            meshes.append(o.data)
    return meshes


class MESH_OT_unsubdivide_seam_aware(_CommonProps, bpy.types.Operator):
    """Un-Subdivide the mesh data directly (keeps shape keys, respects seams)"""
    bl_idname = "mesh.unsubdivide_seam_aware"
    bl_label = "Apply Un-Subdivide"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(o.type == 'MESH' for o in context.selected_objects) or \
            (context.active_object and context.active_object.type == 'MESH')

    def draw(self, context):
        _draw_props(self.layout, self, context)

    def execute(self, context):
        was_edit = context.mode == 'EDIT_MESH'
        if was_edit:
            bpy.ops.object.mode_set(mode='OBJECT')
        opts = _opts_from(self)
        total_b = total_a = 0
        try:
            for me in _target_meshes(context):
                b, a = unsubdivide_mesh(me, self.iterations, opts)
                total_b += b
                total_a += a
        finally:
            if was_edit:
                bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, f"Un-Subdivide: {total_b} → {total_a} vertices "
                              f"({total_b - total_a} removed)")
        return {'FINISHED'}


class MESH_OT_unsubdivide_preview(_CommonProps, bpy.types.Operator):
    """Select the vertices that would be removed (Edit Mode)"""
    bl_idname = "mesh.unsubdivide_preview"
    bl_label = "Select Verts to Remove"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def draw(self, context):
        _draw_props(self.layout, self, context)

    def execute(self, context):
        opts = _opts_from(self)
        count = 0
        for ob in context.objects_in_mode_unique_data:
            if ob.type != 'MESH':
                continue
            bm = bmesh.from_edit_mesh(ob.data)
            bm.verts.index_update()
            work = bm.copy()
            layer = work.verts.layers.int.new(_ORIG_LAYER)
            for i, v in enumerate(work.verts):
                v[layer] = i
            unsubdivide_bmesh(work, self.iterations, opts)
            kept = {v[layer] for v in work.verts}
            work.free()

            for f in bm.faces:
                f.select = False
            for e in bm.edges:
                e.select = False
            for v in bm.verts:
                v.select = v.index not in kept
                count += v.select
            bm.select_flush_mode()
            bmesh.update_edit_mesh(ob.data, loop_triangles=False, destructive=False)
        context.tool_settings.mesh_select_mode = (True, False, False)
        self.report({'INFO'}, f"{count} vertices would be removed")
        return {'FINISHED'}


def _draw_props(layout, s, context):
    col = layout.column()
    col.prop(s, "iterations")
    row = col.row(align=True)
    row.prop(s, "use_mirror")
    sub = row.row(align=True)
    sub.active = s.use_mirror
    sub.prop(s, "mirror_axis", expand=True)
    col.prop(s, "flip_pattern")
    row = col.row(align=True)
    row.prop(s, "use_cleanup")
    sub = row.row(align=True)
    sub.active = s.use_cleanup
    sub.prop(s, "cleanup_angle", text="")
    col.separator()
    col.label(text="Protect:")
    col.prop(s, "respect_seams")
    col.prop(s, "respect_uv_islands")
    col.prop(s, "respect_sharp")
    col.prop(s, "respect_materials")
    sub = col.column()
    sub.active = context.mode == 'EDIT_MESH'
    sub.prop(s, "selected_only")


def _copy_settings(op, s):
    for name in _PROP_NAMES:
        setattr(op, name, getattr(s, name))


class VIEW3D_PT_unsubdivide(bpy.types.Panel):
    bl_label = "Un-Subdivide"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Un-Subdivide"

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def draw(self, context):
        s = context.scene.unsubdiv_settings
        _draw_props(self.layout, s, context)
        col = self.layout.column()
        col.separator()
        if context.mode == 'EDIT_MESH':
            _copy_settings(col.operator(MESH_OT_unsubdivide_preview.bl_idname,
                                        icon='RESTRICT_SELECT_OFF'), s)
        _copy_settings(col.operator(MESH_OT_unsubdivide_seam_aware.bl_idname,
                                    icon='MOD_DECIM'), s)
        me = context.active_object.data
        if me.shape_keys:
            col.label(text=f"{len(me.shape_keys.key_blocks)} shape keys will be kept",
                      icon='SHAPEKEY_DATA')


def _menu(self, context):
    self.layout.operator(MESH_OT_unsubdivide_seam_aware.bl_idname, icon='MOD_DECIM')


classes = (UNSUBDIV_PG_settings, MESH_OT_unsubdivide_seam_aware,
           MESH_OT_unsubdivide_preview, VIEW3D_PT_unsubdivide)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.unsubdiv_settings = PointerProperty(type=UNSUBDIV_PG_settings)
    bpy.types.VIEW3D_MT_edit_mesh_clean.append(_menu)
    bpy.types.VIEW3D_MT_object.append(_menu)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(_menu)
    bpy.types.VIEW3D_MT_edit_mesh_clean.remove(_menu)
    del bpy.types.Scene.unsubdiv_settings
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
