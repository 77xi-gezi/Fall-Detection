"""Start a password-protected HTTPS Quick Tunnel without opening inbound ports."""
import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".tools/public_access.json"
BINARY = ROOT / ".tools/cloudflared.exe"
TUNNEL_LOG = ROOT / ".tools/public_tunnel.log"
WEB_LOG = ROOT / ".tools/public_web.log"


def read_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_state(info: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def owned_process(info: dict, kind: str):
    record = info.get(kind, {})
    try:
        process = psutil.Process(record["pid"])
        if abs(process.create_time() - record["created"]) > 1:
            return None
        if kind == "tunnel":
            matches = Path(process.exe()).resolve() == BINARY.resolve()
        else:
            matches = any(
                arg.replace("\\", "/") == str(ROOT / "web_app.py").replace("\\", "/")
                for arg in process.cmdline()
            )
        return process if matches and process.is_running() else None
    except (KeyError, psutil.Error):
        return None


def stop(info: dict) -> None:
    for kind in ("tunnel", "web"):
        process = owned_process(info, kind)
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                process.kill()
    if info:
        info["running"] = False
        save_state(info)


def ensure_binary() -> None:
    if BINARY.is_file():
        # Installed copies must have a recorded checksum from the official GitHub asset.
        digest_file = BINARY.with_suffix(".sha256")
        if digest_file.is_file():
            expected = digest_file.read_text(encoding="ascii").strip()
            if hashlib.sha256(BINARY.read_bytes()).hexdigest() == expected:
                return
    url = "https://api.github.com/repos/cloudflare/cloudflared/releases/latest"
    request = urllib.request.Request(url, headers={"User-Agent": "Fall-Detection-Setup"})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    asset = next(item for item in release["assets"]
                 if item["name"] == "cloudflared-windows-amd64.exe")
    digest = asset.get("digest", "")
    if not digest.startswith("sha256:"):
        raise RuntimeError("The official release does not contain a SHA256 checksum")
    expected = digest.split(":", 1)[1]
    if not BINARY.is_file() or hashlib.sha256(BINARY.read_bytes()).hexdigest() != expected:
        print("Downloading official Cloudflare tunnel program...", flush=True)
        temporary = BINARY.with_suffix(".download")
        with urllib.request.urlopen(asset["browser_download_url"], timeout=120) as response:
            with temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Cloudflare download checksum mismatch")
        temporary.replace(BINARY)
    BINARY.with_suffix(".sha256").write_text(expected, encoding="ascii")


def spawn(command: list[str], log: Path, env: dict):
    with log.open("wb") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    created = psutil.Process(process.pid).create_time()
    return process, {"pid": process.pid, "created": created}


def public_health(info: dict, port: int) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    encoded = base64.b64encode(f"{info['username']}:{info['password']}".encode()).decode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/health", headers={"Authorization": "Basic " + encoded},
    )
    try:
        with opener.open(request, timeout=5) as response:
            return bool(json.load(response).get("ready"))
    except (OSError, urllib.error.URLError):
        return False


def print_access(info: dict) -> None:
    print("\nPublic URL: " + info["url"], flush=True)
    print("Username: " + info["username"], flush=True)
    print("Password: " + info["password"], flush=True)
    print("Keep this PC, Docker, and the tunnel running. Restarting changes the URL.", flush=True)
    print("Stop external access: double-click stop_public.cmd", flush=True)


def start(port: int) -> dict:
    ROOT.joinpath(".tools").mkdir(parents=True, exist_ok=True)
    info = read_state()
    if (info.get("running") and owned_process(info, "tunnel")
            and owned_process(info, "web") and public_health(info, info.get("port", port))):
        print_access(info)
        return info
    stop(info)
    # Fail before creating a tunnel if this dedicated local port belongs to another app.
    import socket

    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"Local port {port} is in use; choose another --port") from error
    ensure_binary()
    env = os.environ.copy()
    for name, value in urllib.request.getproxies().items():
        if name in ("http", "https"):
            env.setdefault(name.upper() + "_PROXY", value)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["PYTHONUTF8"] = "1"
    env["YOLO_CONFIG_DIR"] = str(ROOT / ".tools/ultralytics")
    env["MPLCONFIGDIR"] = str(ROOT / ".tools/matplotlib")
    info = {"username": "fall", "password": secrets.token_urlsafe(18),
            "port": port, "running": False}
    try:
        tunnel, info["tunnel"] = spawn([
            str(BINARY), "--no-autoupdate", "tunnel", "--url", f"http://127.0.0.1:{port}",
            "--protocol", "http2", "--loglevel", "info",
        ], TUNNEL_LOG, env)
        save_state(info)
        deadline = time.monotonic() + 65
        while time.monotonic() < deadline:
            log = TUNNEL_LOG.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
            if match:
                info["url"] = match.group()
                break
            if tunnel.poll() is not None:
                raise RuntimeError("Tunnel exited: " + log[-1800:])
            time.sleep(0.5)
        if "url" not in info:
            raise RuntimeError("No public URL was created. Check .tools/public_tunnel.log")
        save_state(info)
        web, info["web"] = spawn([
            sys.executable, "-X", "utf8", str(ROOT / "web_app.py"), "--port", str(port),
            "--public-url", info["url"], "--access-file", str(STATE),
        ], WEB_LOG, env)
        save_state(info)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if public_health(info, port):
                info["running"] = True
                save_state(info)
                print_access(info)
                return info
            if web.poll() is not None:
                raise RuntimeError("Public web server exited. Check .tools/public_web.log")
            time.sleep(0.5)
        raise RuntimeError("Public web service is not ready. Check Docker and model readiness.")
    except BaseException:
        stop(info)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Password-protected external browser access")
    parser.add_argument("action", choices=("start", "stop", "status"), nargs="?", default="start")
    parser.add_argument("--port", type=int, default=8081)
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        if arguments.action == "stop":
            stop(read_state())
            print("External access stopped. The local 8080 web service is unchanged.")
        elif arguments.action == "status":
            info = read_state()
            if info.get("running") and owned_process(info, "tunnel") and owned_process(info, "web"):
                print_access(info)
            else:
                print("External access is not running.")
                return 1
        else:
            start(arguments.port)
        return 0
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print("Public access failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
