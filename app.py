import os
import re
import base64
import time
import json
import shutil
import subprocess
import tempfile
import threading
import queue
import uuid
import struct
import requests as http_requests
import boto3
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError
from flask import Flask, request, send_file, jsonify, Response, stream_with_context
import cascadio
import pygltflib
from dotenv import load_dotenv

from clean_normals import clean_glb as _clean_glb

load_dotenv()


# ---------------------------------------------------------------------------
# GLB normal computation
# ---------------------------------------------------------------------------

_COMPONENT_TYPE_TO_DTYPE = {
    pygltflib.BYTE:           np.int8,
    pygltflib.UNSIGNED_BYTE:  np.uint8,
    pygltflib.SHORT:          np.int16,
    pygltflib.UNSIGNED_SHORT: np.uint16,
    pygltflib.UNSIGNED_INT:   np.uint32,
    pygltflib.FLOAT:          np.float32,
}
_TYPE_TO_COMPONENTS = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4, "MAT3": 9, "MAT4": 16,
}


def _accessor_to_array(glb: pygltflib.GLTF2, accessor_idx: int) -> np.ndarray:
    """Read a glTF accessor into a numpy array."""
    acc   = glb.accessors[accessor_idx]
    bview = glb.bufferViews[acc.bufferView]
    blob  = glb.binary_blob()
    dtype = _COMPONENT_TYPE_TO_DTYPE[acc.componentType]
    n_comp = _TYPE_TO_COMPONENTS[acc.type]
    byte_offset = (bview.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count
    data = np.frombuffer(blob, dtype=dtype,
                         count=count * n_comp,
                         offset=byte_offset)
    return data.reshape(count, n_comp) if n_comp > 1 else data


def _sanitize_glb_names(glb: pygltflib.GLTF2) -> int:
    """
    Replace every node/mesh/scene name in the GLB with a Houdini-safe variant
    (alphanumeric + underscore only, never empty, must not start with a digit).
    Returns the number of names changed.
    """
    safe_re = re.compile(r"[^A-Za-z0-9_]+")
    changed = 0
    seen = set()

    def _safe(name, fallback):
        if not name:
            base = fallback
        else:
            base = safe_re.sub("_", name).strip("_") or fallback
            if base[0].isdigit():
                base = f"_{base}"
        # Disambiguate duplicates
        candidate = base
        i = 1
        while candidate in seen:
            i += 1
            candidate = f"{base}_{i}"
        seen.add(candidate)
        return candidate

    for i, scene in enumerate(glb.scenes or []):
        new = _safe(scene.name, f"scene_{i}")
        if new != scene.name:
            scene.name = new
            changed += 1
    seen.clear()
    for i, node in enumerate(glb.nodes or []):
        new = _safe(node.name, f"node_{i}")
        if new != node.name:
            node.name = new
            changed += 1
    seen.clear()
    for i, mesh in enumerate(glb.meshes or []):
        new = _safe(mesh.name, f"mesh_{i}")
        if new != mesh.name:
            mesh.name = new
            changed += 1
    return changed


def add_smooth_normals(glb_path: str, crease_angle_deg: float = 30.0) -> int:
    """
    Post-process a GLB file in-place. See module docstring above.
    Returns the total triangle count across all primitives.
    """
    glb = pygltflib.GLTF2().load(glb_path)
    blob = bytearray(glb.binary_blob())

    n_renamed = _sanitize_glb_names(glb)

    new_accessors  = list(glb.accessors)
    new_buffer_views = list(glb.bufferViews)

    cos_crease = float(np.cos(np.deg2rad(crease_angle_deg)))

    t0 = time.time()
    n_prims_done = 0
    n_sharp_total = 0
    n_tris_total  = 0
    for mesh in glb.meshes:
        for prim in mesh.primitives:
            if prim.attributes.POSITION is None:
                continue

            pos = _accessor_to_array(glb, prim.attributes.POSITION).astype(np.float32)

            if prim.indices is not None:
                idx = _accessor_to_array(glb, prim.indices).astype(np.int32).ravel()
            else:
                idx = np.arange(len(pos), dtype=np.int32)

            n_verts = len(pos)
            triangles = idx.reshape(-1, 3)
            n_tris = len(triangles)
            n_tris_total += n_tris

            # ---- Weld coincident vertices for normal accumulation ------------
            _, welded_idx = np.unique(pos, axis=0, return_inverse=True)
            welded_idx = welded_idx.astype(np.int32)
            n_welded = int(welded_idx.max()) + 1

            tri_welded = welded_idx[triangles]

            # Per-face normals (length = 2 * triangle area)
            v0 = pos[triangles[:, 0]]
            v1 = pos[triangles[:, 1]]
            v2 = pos[triangles[:, 2]]
            face_normals = np.cross(v1 - v0, v2 - v0)
            fn_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
            fn_len_safe = np.where(fn_len < 1e-12, 1.0, fn_len)
            fn_unit = (face_normals / fn_len_safe).astype(np.float32)  # (n_tris, 3)

            # Per-corner angle weight × face normal
            corners = np.empty((3, n_tris, 3), dtype=np.float32)
            for i in range(3):
                a = pos[triangles[:, (i + 1) % 3]] - pos[triangles[:, i]]
                b = pos[triangles[:, (i + 2) % 3]] - pos[triangles[:, i]]
                a_len = np.linalg.norm(a, axis=1)
                b_len = np.linalg.norm(b, axis=1)
                denom = np.where((a_len < 1e-12) | (b_len < 1e-12),
                                 1.0, a_len * b_len)
                cos_a = np.clip(np.einsum("ij,ij->i", a, b) / denom, -1.0, 1.0)
                corners[i] = (np.arccos(cos_a)[:, None] * fn_unit).astype(np.float32)

            # Scatter-add into the welded vertex space
            welded_normals = np.zeros((n_welded, 3), dtype=np.float64)
            for i in range(3):
                np.add.at(welded_normals, tri_welded[:, i], corners[i])

            wn = np.linalg.norm(welded_normals, axis=1, keepdims=True)
            wn = np.where(wn < 1e-12, 1.0, wn)
            welded_normals = (welded_normals / wn).astype(np.float32)

            # Splat back to every original vertex
            smooth_per_vertex = welded_normals[welded_idx]  # (n_verts, 3)

            # ---- Crease-aware fallback ---------------------------------------
            # Each original vertex belongs to exactly one triangle (cascadio
            # splits per corner). Build a per-vertex face-normal vector and
            # compare it with the smoothed normal. If the angle exceeds the
            # crease threshold, this corner is on a hard edge → use the face
            # normal directly.
            face_n_per_vertex = np.empty((n_verts, 3), dtype=np.float32)
            face_n_per_vertex[triangles[:, 0]] = fn_unit
            face_n_per_vertex[triangles[:, 1]] = fn_unit
            face_n_per_vertex[triangles[:, 2]] = fn_unit

            dots = np.einsum("ij,ij->i", smooth_per_vertex, face_n_per_vertex)
            sharp_mask = dots < cos_crease  # True where smoothing crosses a crease

            vertex_normals = smooth_per_vertex.copy()
            vertex_normals[sharp_mask] = face_n_per_vertex[sharp_mask]

            # Re-normalise (defensive — splatted/face normals are unit, but
            # numerical drift in the splat is possible)
            vn = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            vn = np.where(vn < 1e-12, 1.0, vn)
            vertex_normals = (vertex_normals / vn).astype(np.float32)

            n_sharp_total += int(sharp_mask.sum())

            # Append normal data to the binary blob
            normal_bytes = vertex_normals.tobytes()
            byte_offset  = len(blob)
            blob.extend(normal_bytes)

            bv_idx = len(new_buffer_views)
            new_buffer_views.append(pygltflib.BufferView(
                buffer=0,
                byteOffset=byte_offset,
                byteLength=len(normal_bytes),
                target=pygltflib.ARRAY_BUFFER,
            ))

            acc_idx = len(new_accessors)
            mins = vertex_normals.min(axis=0).tolist()
            maxs = vertex_normals.max(axis=0).tolist()
            new_accessors.append(pygltflib.Accessor(
                bufferView=bv_idx,
                byteOffset=0,
                componentType=pygltflib.FLOAT,
                count=n_verts,
                type="VEC3",
                min=mins,
                max=maxs,
            ))

            prim.attributes.NORMAL = acc_idx
            n_prims_done += 1

    glb.accessors   = new_accessors
    glb.bufferViews = new_buffer_views

    # Update buffer byte length
    glb.buffers[0].byteLength = len(blob)
    glb.set_binary_blob(bytes(blob))
    glb.save(glb_path)
    elapsed = time.time() - t0
    print(f"[normals] ✓ {n_prims_done} primitive(s), {n_tris_total} tri(s) "
          f"processed in {elapsed:.2f}s "
          f"(crease={crease_angle_deg}°, {n_sharp_total} sharp corner(s); "
          f"{n_renamed} name(s) sanitized) → {glb_path}",
          flush=True)
    return n_tris_total


app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def index():
    return app.send_static_file("index.html")


# ---------------------------------------------------------------------------
# Async conversion jobs with live stderr streaming via SSE
# ---------------------------------------------------------------------------

# Job registry: { job_id: {status, lines: queue.Queue, result_path, work_dir,
#                          error, stem, t_start} }
_JOBS = {}
_JOBS_LOCK = threading.Lock()


def _emit(job, kind, payload=None):
    """Push an event onto the job's SSE queue."""
    job["lines"].put({"kind": kind, "payload": payload})


def _capture_fd_to_queue(read_fd, job, source_label):
    """Read from a pipe fd line-by-line and push events to the job queue."""
    try:
        with os.fdopen(read_fd, "r", buffering=1, errors="replace") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                # Mirror to server stdout for debugging
                print(f"[{source_label}] {line}", flush=True)
                _emit(job, "log", {"source": source_label, "line": line})
    except Exception as e:
        _emit(job, "log",
              {"source": "capture", "line": f"(stderr capture error: {e})"})


def _run_conversion(job_id, input_path, output_path, params, stem):
    """Worker: redirect fd 1 & 2 to pipes, run cascadio + normals, push events."""
    job = _JOBS[job_id]
    job["status"] = "running"
    t_start = time.time()
    _emit(job, "stage",
          {"name": "cascadio", "message": "Lancement de la tessellation..."})

    # Pipe for stderr (fd 2) of native code — that's where OpenCASCADE writes
    # its messages like "*** ERR StepReaderData ***". We do NOT redirect fd 1
    # because our own print() calls inside the capture thread would loop back
    # into the pipe.
    r_err, w_err = os.pipe()
    saved_stderr = os.dup(2)
    os.dup2(w_err, 2)
    os.close(w_err)

    t_err = threading.Thread(
        target=_capture_fd_to_queue, args=(r_err, job, "cascadio"),
        daemon=True)
    t_err.start()

    code = None
    crash = None
    try:
        try:
            code = cascadio.step_to_glb(
                input_path, output_path,
                tol_linear=params["tol_linear"],
                tol_angular=params["tol_angular"],
                tol_relative=params["tol_relative"],
            )
        except Exception as e:
            crash = f"{type(e).__name__}: {e}"
    finally:
        # Restore fd 2 (closes the write end of our pipe → reader sees EOF)
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)

    t_err.join(timeout=2)

    cascadio_elapsed = time.time() - t_start

    if crash is not None:
        job["error"] = f"cascadio crashed: {crash}"
        _emit(job, "error", {"message": job["error"]})
        _emit(job, "end", None)
        job["status"] = "error"
        return

    if code != 0:
        job["error"] = f"cascadio returned code {code}"
        _emit(job, "error", {"message": job["error"]})
        _emit(job, "end", None)
        job["status"] = "error"
        return

    if not os.path.exists(output_path):
        job["error"] = "cascadio produced no output file"
        _emit(job, "error", {"message": job["error"]})
        _emit(job, "end", None)
        job["status"] = "error"
        return

    out_size = os.path.getsize(output_path)
    _emit(job, "stage", {
        "name": "cascadio_done",
        "message": f"cascadio OK en {cascadio_elapsed:.2f}s "
                   f"({out_size/1024:.1f} KB)",
    })

    # Normals post-processing
    crease_angle = float(params.get("crease_angle", 30.0))
    _emit(job, "stage",
          {"name": "normals",
           "message": f"Calcul des normales (crease={crease_angle:g}°)..."})
    t_n = time.time()
    n_tris_total = 0
    try:
        n_tris_total = add_smooth_normals(
            output_path, crease_angle_deg=crease_angle)
        _emit(job, "stage", {
            "name": "normals_done",
            "message": f"Normales injectées en {time.time() - t_n:.2f}s "
                       f"({n_tris_total} triangles)",
        })
    except Exception as e:
        _emit(job, "log", {
            "source": "normals",
            "line": f"⚠ add_smooth_normals a échoué : "
                    f"{type(e).__name__}: {e} (GLB brut conservé)",
        })

    job["result_path"] = output_path
    job["status"] = "done"
    total = time.time() - t_start
    _emit(job, "done", {
        "message": f"Conversion terminée en {total:.2f}s",
        "stem": stem,
        "size_kb": round(out_size / 1024, 1),
        "triangles": n_tris_total,
        "quality": params.get("quality"),
    })
    _emit(job, "end", None)


@app.route("/convert", methods=["POST"])
def convert():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith((".step", ".stp")):
        return jsonify({"error": "File must be a STEP file (.step or .stp)"}), 400

    # Optional tessellation tolerances from the client
    def _to_float(name, default):
        try:
            return float(request.form.get(name, default))
        except (TypeError, ValueError):
            return default

    params = {
        "tol_linear":   _to_float("tol_linear", 0.01),
        "tol_angular":  _to_float("tol_angular", 0.5),
        "tol_relative": (request.form.get("tol_relative", "false").lower()
                         in ("1", "true", "yes", "on")),
        "crease_angle": _to_float("crease_angle", 30.0),
        "quality":      int(_to_float("quality", 3)),
    }

    # Persist file in a per-job temp dir (cleaned on result fetch)
    work_dir = tempfile.mkdtemp(prefix="cad_job_")
    input_path  = os.path.join(work_dir, f"{uuid.uuid4()}.step")
    output_path = os.path.join(work_dir, f"{uuid.uuid4()}.glb")
    file.save(input_path)
    in_size = os.path.getsize(input_path)

    job_id = uuid.uuid4().hex
    stem   = os.path.splitext(file.filename)[0]

    job = {
        "status":      "queued",
        "lines":       queue.Queue(),
        "result_path": None,
        "work_dir":    work_dir,
        "error":       None,
        "stem":        stem,
        "t_start":     time.time(),
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    print(f"[convert] ▶ job {job_id} : {file.filename} "
          f"({in_size/1024:.1f} KB), params={params}", flush=True)

    _emit(job, "stage", {
        "name": "queued",
        "message": f"{file.filename} ({in_size/1024:.1f} KB) en file...",
    })

    t = threading.Thread(
        target=_run_conversion,
        args=(job_id, input_path, output_path, params, stem),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/convert/stream/<job_id>")
def convert_stream(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404

    @stream_with_context
    def gen():
        # Heartbeat every ~15s to keep the connection alive
        while True:
            try:
                ev = job["lines"].get(timeout=15)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(ev)}\n\n"
            if ev["kind"] == "end":
                break

    headers = {
        "Content-Type":      "text/event-stream",
        "Cache-Control":     "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection":        "keep-alive",
    }
    return Response(gen(), headers=headers)


@app.route("/convert/result/<job_id>")
def convert_result(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    if job["status"] != "done" or not job["result_path"]:
        return jsonify({"error": f"job not ready (status={job['status']})"}), 409

    result_path = job["result_path"]
    work_dir    = job["work_dir"]
    stem        = job["stem"]

    @stream_with_context
    def stream_file():
        try:
            with open(result_path, "rb") as f:
                while True:
                    chunk = f.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk
        finally:
            # Clean up after streaming
            with _JOBS_LOCK:
                _JOBS.pop(job_id, None)
            shutil.rmtree(work_dir, ignore_errors=True)

    headers = {
        "Content-Type":        "model/gltf-binary",
        "Content-Disposition": f'attachment; filename="{stem}.glb"',
    }
    return Response(stream_file(), headers=headers)


# ---------------------------------------------------------------------------
# Web optimization pipeline (clean_normals → gltfpack)
# ---------------------------------------------------------------------------

# Quality presets for the "Optimize for Web" pass.
# Each entry maps a 1-5 slider value to a list of gltfpack flags.
# Common flags applied to every level:
#   -kn  keep normals (preserve hard edges & shading)
#   -km  keep mesh names (LOD / picking)
#   -ke  keep extras (UV maps, custom data)
WEB_PRESETS = {
    1: {"label": "Lossless",  "flags": ["-noq"]},
    2: {"label": "Léger",     "flags": ["-si", "0.7", "-cc"]},
    3: {"label": "Standard",  "flags": ["-si", "0.5", "-mi", "-cc"]},
    4: {"label": "Compact",   "flags": ["-si", "0.2", "-mi", "-cc",
                                          "-vp", "12", "-vt", "10", "-vn", "8"]},
    5: {"label": "Aggressif", "flags": ["-si", "0.05", "-mi", "-cc",
                                          "-vp", "11", "-vt", "8",  "-vn", "6"]},
}
_WEB_COMMON = ["-kn", "-km", "-ke"]


def _glb_triangle_count(path):
    """Best-effort triangle count from a GLB by reading accessor counts only.
    Safe on quantized GLBs (no buffer parsing)."""
    try:
        from pygltflib import GLTF2
        gltf = GLTF2().load(path)
        n = 0
        for mesh in (gltf.meshes or []):
            for prim in (mesh.primitives or []):
                if prim.indices is not None:
                    n += gltf.accessors[prim.indices].count // 3
                elif prim.attributes and prim.attributes.POSITION is not None:
                    n += gltf.accessors[prim.attributes.POSITION].count // 3
        return n
    except Exception:
        return None


def _run_cmd_streamed(job, cmd, label):
    """Run a subprocess streaming combined stdout/stderr into the job queue."""
    _emit(job, "log", {"source": label, "line": f"$ {' '.join(cmd)}"})
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"`{cmd[0]}` introuvable dans le PATH. Installe-le "
            f"(brew install gltfpack ou npm i -g gltfpack) puis relance."
        ) from e

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\r\n")
        if line:
            print(f"[{label}] {line}", flush=True)
            _emit(job, "log", {"source": label, "line": line})
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{cmd[0]} returned code {code}")


def _run_optimization(job_id, input_glb_path, stem, quality=3):
    """Worker: clean_normals (incl. node→mesh names) → gltfpack with preset.

    `quality` (1-5) selects a preset from WEB_PRESETS. See the table at the top
    of this section for what each level does.
    """
    job = _JOBS[job_id]
    job["status"] = "running"
    t_start = time.time()

    work_dir   = job["work_dir"]
    cleaned    = os.path.join(work_dir, "cleaned.glb")
    web_final  = os.path.join(work_dir, f"{stem}_web.glb")

    preset = WEB_PRESETS.get(int(quality), WEB_PRESETS[3])
    flags  = preset["flags"] + _WEB_COMMON

    try:
        # 1. clean_normals (also propagates node.name → mesh.name)
        _emit(job, "stage",
              {"name": "clean",
               "message": "Nettoyage des normales et propagation des noms..."})
        _clean_glb(input_glb_path, cleaned)
        _emit(job, "log", {
            "source": "clean",
            "line": f"cleaned.glb : {os.path.getsize(cleaned)/1024:.1f} KB",
        })

        # 2. gltfpack — preset-driven
        _emit(job, "stage",
              {"name": "gltfpack",
               "message": f"gltfpack [{preset['label']}]..."})
        _run_cmd_streamed(
            job,
            ["gltfpack", "-i", cleaned, "-o", web_final, *flags],
            "gltfpack",
        )
        _emit(job, "log", {
            "source": "gltfpack",
            "line": f"{stem}_web.glb : {os.path.getsize(web_final)/1024:.1f} KB",
        })

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        job["error"] = msg
        _emit(job, "error", {"message": msg})
        _emit(job, "end", None)
        job["status"] = "error"
        return

    job["result_path"] = web_final
    job["status"] = "done"
    out_size = os.path.getsize(web_final)
    triangles = _glb_triangle_count(web_final)
    total = time.time() - t_start
    _emit(job, "done", {
        "message": f"Optimisation terminée en {total:.2f}s",
        "stem": f"{stem}_web",
        "size_kb": round(out_size / 1024, 1),
        "triangles": triangles,
        "preset": preset["label"],
        "quality": int(quality),
    })
    _emit(job, "end", None)


@app.route("/optimize", methods=["POST"])
def optimize():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".glb"):
        return jsonify({"error": "File must be a .glb"}), 400

    try:
        quality = int(request.form.get("quality", "3"))
    except (TypeError, ValueError):
        quality = 3
    quality = max(1, min(5, quality))

    work_dir   = tempfile.mkdtemp(prefix="cad_opt_")
    input_path = os.path.join(work_dir, "input.glb")
    file.save(input_path)
    in_size = os.path.getsize(input_path)

    job_id = uuid.uuid4().hex
    stem   = os.path.splitext(os.path.basename(file.filename))[0]
    # Trim a trailing "_glb" if the client appended it
    stem   = re.sub(r"_glb$", "", stem)

    job = {
        "status":      "queued",
        "lines":       queue.Queue(),
        "result_path": None,
        "work_dir":    work_dir,
        "error":       None,
        "stem":        stem,
        "t_start":     time.time(),
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    preset_label = WEB_PRESETS[quality]["label"]
    print(f"[optimize] ▶ job {job_id} : {file.filename} "
          f"({in_size/1024:.1f} KB, preset {quality}/{preset_label})", flush=True)

    _emit(job, "stage", {
        "name": "queued",
        "message": f"{file.filename} ({in_size/1024:.1f} KB) "
                   f"— preset {preset_label}",
    })

    t = threading.Thread(
        target=_run_optimization,
        args=(job_id, input_path, stem, quality),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})



VIEWS_DIR = os.path.join(os.path.dirname(__file__), "views")


@app.route("/save-views", methods=["POST"])
def save_views():
    data = request.get_json(force=True)
    stem   = data.get("stem", "model") or "model"
    images = data.get("images", {})

    # Sanitise stem to prevent path traversal
    stem = os.path.basename(stem).replace("..", "").strip() or "model"

    dest = os.path.join(VIEWS_DIR, stem)
    os.makedirs(dest, exist_ok=True)

    saved = 0
    for view_name, b64 in images.items():
        safe_name = os.path.basename(view_name).replace("..", "") + ".png"
        with open(os.path.join(dest, safe_name), "wb") as f:
            f.write(base64.b64decode(b64))
        saved += 1

    token = fetch_hitem3d_token()
    return jsonify({"saved": saved, "folder": dest, "token": token})


def fetch_hitem3d_token():
    client_id     = os.environ.get("ACCESS_KEY", "")
    client_secret = os.environ.get("SECRET_KEY", "")

    if not client_id or not client_secret:
        print("❌ ACCESS_KEY ou SECRET_KEY manquant dans .env")
        return None

    credentials = base64.b64encode(
        f"{client_id}:{client_secret}".encode("utf-8")
    ).decode("utf-8")

    try:
        resp = http_requests.post(
            "https://api.hitem3d.ai/open-api/v1/auth/token",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type":  "application/json",
                "Accept":        "*/*",
            },
            json={},
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json().get("token") or resp.json().get("access_token") or resp.text
        print(f"✓ Token : {token}")
        return token
    except http_requests.exceptions.HTTPError as e:
        print(f"❌ HTTP {e.response.status_code} : {e.response.text}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
    return None


# --------------------------------------------------------------------------
# Texture enhancement via Qwen image-edit API
# --------------------------------------------------------------------------

_INFER_BASE  = "https://staging-inference-engine.api-chat3d.com/v1"
_S3_BUCKET   = os.environ.get("S3_BUCKET", "andrea-988441895596-eu-west-3")
_S3_PREFIX   = "textureinfer/img_in/"
_BASE_PROMPT = (
    "This is a render of a 3D object shown in flat grey (no texture). "
    "Your task is to apply realistic textures and colors to this object while "
    "preserving its exact geometry, topology, proportions, and camera angle. "
    "Do NOT modify the shape, silhouette, or structure of the object in any way. "
    "Apply the textures consistently across the entire visible surface. "
    "Follow these instructions from the user to texture the object: "
)
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "views", "generated")


def _s3_upload_and_presign(local_path: str, key: str, expires: int = 3600) -> str:
    """Upload file to S3 and return a presigned GET URL valid for `expires` seconds."""
    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("AWS_REGION", "eu-west-3"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    s3.upload_file(local_path, _S3_BUCKET, key, ExtraArgs={"ContentType": "image/png"})
    region = os.environ.get("AWS_REGION", "eu-west-3")
    url = f"https://{_S3_BUCKET}.s3.{region}.amazonaws.com/{key}"
    return url


@app.route("/apply-texture", methods=["POST"])
def apply_texture():
    data   = request.get_json(force=True)
    stem   = os.path.basename((data.get("stem") or "model").replace("..", "")).strip() or "model"
    prompt = (data.get("prompt") or "").strip()
    token  = (data.get("token") or "").strip()

    if not prompt:
        return jsonify({"error": "prompt requis"}), 400
    if not token:
        return jsonify({"error": "token manquant"}), 400

    src_dir  = os.path.join(VIEWS_DIR, stem)
    dest_dir = os.path.join(GENERATED_DIR, stem)
    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.isdir(src_dir):
        return jsonify({"error": f"Dossier source introuvable : {src_dir}"}), 404

    all_png = sorted(p for p in os.listdir(src_dir) if p.lower().endswith(".png"))
    if not all_png:
        return jsonify({"error": "Aucune image PNG dans le dossier source"}), 404
    # Only process the quarter_right view for Qwen
    png_files = [p for p in all_png if "quarter_right" in p] or all_png[:1]

    full_prompt = _BASE_PROMPT + prompt

    infer_token = os.environ.get("INFERENCE_API_TOKEN") or os.environ.get("UNIFIED_INFERENCE_API_KEY", "")
    if not infer_token:
        return jsonify({"error": "INFERENCE_API_TOKEN manquant dans .env"}), 500

    infer_headers = {
        "Accept":        "application/json",
        "Authorization": f"Bearer {infer_token}",
        "Content-Type":  "application/json",
    }
    poll_headers = {
        "accept":        "application/json",
        "Authorization": f"Bearer {infer_token}",
    }

    results = []
    for fname in png_files:
        img_path = os.path.join(src_dir, fname)

        # Sanitise stem and fname for S3 key (no spaces, no special chars)
        safe_stem  = re.sub(r"[^\w\-]", "_", stem)
        safe_fname = re.sub(r"[^\w\-.]", "_", fname)
        s3_key = f"{_S3_PREFIX}{safe_stem}/{uuid.uuid4().hex}_{safe_fname}"
        try:
            img_url = _s3_upload_and_presign(img_path, s3_key)
            print(f"[texture] ☁ {fname} → s3://{_S3_BUCKET}/{s3_key}")
        except (BotoCoreError, ClientError, Exception) as e:
            print(f"[texture] ❌ S3 upload {fname}: {e}")
            results.append({"view": fname, "status": "error", "error": f"S3 upload: {e}"})
            continue

        payload = {
            "version": "replicate/qwen-image-edit-plus",
            "input": {
                "image":  [img_url],
                "prompt": full_prompt,
            },
        }

        # Submit job
        try:
            resp = http_requests.post(
                f"{_INFER_BASE}/infer",
                headers=infer_headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"[texture] ❌ submit {fname}: {e}")
            results.append({"view": fname, "status": "error", "error": str(e)})
            continue

        job_id = resp.json().get("job_id")
        if not job_id:
            err = f"Pas de job_id: {resp.text}"
            print(f"[texture] ❌ {fname}: {err}")
            results.append({"view": fname, "status": "error", "error": err})
            continue

        print(f"[texture] ⏳ {fname} → job {job_id}")

        # Poll until done
        poll_url = f"{_INFER_BASE}/jobs/{job_id}"
        status = ""
        result_data = {}
        for _ in range(120):          # max ~6 min
            time.sleep(3)
            try:
                pr = http_requests.get(poll_url, headers=poll_headers, timeout=15)
                pr.raise_for_status()
                result_data = pr.json()
            except Exception as e:
                print(f"[texture] ❌ poll {job_id}: {e}")
                break

            status = result_data.get("status", "")
            pct    = result_data.get("progress", 0)
            print(f"[texture] {fname} status={status} {pct}%")
            if status == "SUCCEEDED":
                break
            if status == "FAILED":
                break

        if status != "SUCCEEDED":
            err = result_data.get("error") or result_data.get("message") or status
            print(f"[texture] ❌ {fname} failed: {err}")
            results.append({"view": fname, "status": "error", "error": str(err)})
            continue

        # Download result image
        outputs = result_data.get("outputs", [])
        out_url = (outputs[0] if isinstance(outputs[0], str) else outputs[0].get("url")) if outputs else None
        if not out_url:
            err = "URL de sortie manquante"
            print(f"[texture] ❌ {fname}: {err}")
            results.append({"view": fname, "status": "error", "error": err})
            continue

        try:
            img_resp = http_requests.get(out_url, timeout=60)
            img_resp.raise_for_status()
            out_path = os.path.join(dest_dir, fname)
            with open(out_path, "wb") as f:
                f.write(img_resp.content)
            print(f"[texture] ✓ {fname} → {out_path}")
            results.append({"view": fname, "status": "ok"})
        except Exception as e:
            print(f"[texture] ❌ download {fname}: {e}")
            results.append({"view": fname, "status": "error", "error": str(e)})

    ok_count = sum(1 for r in results if r["status"] == "ok")
    return jsonify({"results": results, "generated": ok_count, "folder": dest_dir})


# --------------------------------------------------------------------------
# hitem3d 3D generation pipeline
# --------------------------------------------------------------------------

_HITEM_API  = "https://api.hitem3d.ai/open-api/v1"
_HITEM_MODEL = "hitem3dv1.5"    # v1.5 needed for request_type=2

# We send 4 views in hitem3d order: front, back, left, right
# mapped from our capture names.
_HITEM_VIEW_MAP = [
    ("quarter_right", "front"),    # slot 0 – front
    ("back",          "back"),     # slot 1 – back
    ("quarter_left",  "left"),     # slot 2 – left
    ("under",         "right"),    # slot 3 – right
]


def _hitem_token():
    """Get a fresh hitem3d Bearer token."""
    cid = os.environ.get("ACCESS_KEY", "")
    cse = os.environ.get("SECRET_KEY", "")
    creds = base64.b64encode(f"{cid}:{cse}".encode()).decode()
    r = http_requests.post(
        f"{_HITEM_API}/auth/token",
        headers={"Authorization": f"Basic {creds}", "Content-Type": "application/json"},
        json={}, timeout=15,
    )
    r.raise_for_status()
    body = r.json()
    token = body.get("data", {}).get("accessToken")
    if not token:
        raise RuntimeError(f"No accessToken in response: {body}")
    print("✓ Token obtained")
    return token


def _hitem_submit(token, files, data):
    r = http_requests.post(
        f"{_HITEM_API}/submit-task",
        headers={"Authorization": f"Bearer {token}"},
        files=files, data=data, timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 200:
        raise RuntimeError(f"hitem3d submit error: {body}")
    tid = body["data"]["task_id"]
    print(f"⏳ task_id={tid}")
    return tid


def _hitem_poll(token, task_id, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = http_requests.get(
            f"{_HITEM_API}/query-task",
            headers={"Authorization": f"Bearer {token}"},
            params={"task_id": task_id}, timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        state = data.get("state", "")
        print(f"poll {task_id}: state={state}")
        if state == "success":
            return data
        if state == "failed":
            raise RuntimeError(f"hitem3d task failed: {data}")
        time.sleep(5)
    raise TimeoutError(f"hitem3d task {task_id} did not complete in {timeout}s")


@app.route("/generate-3d", methods=["POST"])
def generate_3d():
    """Send a single view (quarter_left) → geometry GLB."""
    data = request.get_json(force=True)
    stem = os.path.basename((data.get("stem") or "model").replace("..", "")).strip() or "model"

    src_dir = os.path.join(VIEWS_DIR, stem)
    if not os.path.isdir(src_dir):
        return jsonify({"error": f"Dossier introuvable : {src_dir}"}), 404

    img_path = os.path.join(src_dir, "quarter_left.png")
    if not os.path.exists(img_path):
        return jsonify({"error": "quarter_left.png introuvable"}), 404

    fobj = open(img_path, "rb")
    try:
        token = _hitem_token()

        multi_images = [("multi_images", ("quarter_left.png", fobj, "image/png"))]
        print(f"Stage 1: quarter_left → geometry (bit=1000)")
        task_data = {
            "request_type": "1",
            "model": _HITEM_MODEL,
            "resolution": "512",
            "face": "100000",
            "format": "2",
            "multi_images_bit": "1000",
        }
        task_id = _hitem_submit(token, multi_images, task_data)
        result = _hitem_poll(token, task_id)

        glb_url = result.get("url")
        if not glb_url:
            raise RuntimeError(f"No GLB URL in result: {result}")
        print(f"✓ Geometry: {glb_url}")

        out_dir = os.path.join(VIEWS_DIR, "hitem3d_output", stem)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "geometry.glb")
        dl = http_requests.get(glb_url, timeout=120)
        dl.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(dl.content)
        print(f"✅ Geometry → {out_path}")
        return jsonify({"status": "ok", "glb_url": glb_url, "local_path": out_path})

    except Exception as e:
        print(f"❌ {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        fobj.close()


@app.route("/hitem3d-glb/<path:filename>")
def serve_hitem3d_glb(filename):
    """Serve generated GLB files."""
    safe = os.path.normpath(filename)
    if ".." in safe:
        return jsonify({"error": "invalid path"}), 400
    path = os.path.join(VIEWS_DIR, "hitem3d_output", safe)
    if not os.path.isfile(path):
        return jsonify({"error": "file not found"}), 404
    return send_file(path, mimetype="model/gltf-binary")


if __name__ == "__main__":
    app.run(debug=True, port=8080)

