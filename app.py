import os
import re
import base64
import time
import tempfile
import uuid
import requests as http_requests
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from flask import Flask, request, send_file, jsonify
import cascadio
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith((".step", ".stp")):
        return jsonify({"error": "File must be a STEP file (.step or .stp)"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"{uuid.uuid4()}.step")
        output_path = os.path.join(tmpdir, f"{uuid.uuid4()}.glb")

        file.save(input_path)

        result = cascadio.step_to_glb(input_path, output_path)
        if result != 0:
            return jsonify({"error": f"Conversion failed (code {result})"}), 500

        stem = os.path.splitext(file.filename)[0]
        return send_file(
            output_path,
            mimetype="model/gltf-binary",
            as_attachment=True,
            download_name=f"{stem}.glb",
        )


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

