"""Replay a video through the workcell's full-frame YOLO-Pose and PnP pipeline."""

import argparse
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from twin_depot.cube_completion import Completion
from twin_depot.cube_observer import FastObserver
from twin_depot.cube_pose import camera_matrix


def benchmark_stream(cap, config, model, fps):
    """Drive the normal camera and dashboard HTTP routes on an isolated loopback port."""
    from twin_depot.robot_view import Workcell, desktop_handler, run_worker
    from twin_depot.workcell_source import LatestFrames

    def request(conn, method, path, headers, body=None):
        conn.request(method, path, body, headers)
        response = conn.getresponse()
        value = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(value)
        return value

    def jpeg(image):
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 83])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return data.tobytes()

    ok, frame = cap.read()
    if not ok:
        raise ValueError("Video has no decodable frames")
    with tempfile.TemporaryDirectory() as tmp:
        cell = Workcell(LatestFrames(), config or Path(tmp) / "workcell.json", {}, model.device)
        if "capture_profile" in cell.config.get("intrinsics", {}):
            # A video file has no browser camera identity. Work on the in-memory replay copy.
            cell.config["intrinsics"].pop("capture_profile")
            print("Using supplied intrinsics for replay; verify the recording's lens and crop.")
        cell.pipeline, cell.frame_mode = "fast", "camera"
        if isinstance(model, Completion):
            cell.completion = model
        else:
            cell.fast_observer = model
        handler = desktop_handler(cell, "replay", "12345678")
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server, ThreadPoolExecutor(3) as pool:
            stop = threading.Event()
            pool.submit(server.serve_forever)
            pool.submit(run_worker, cell, None, stop)
            upload = HTTPConnection("127.0.0.1", server.server_port, timeout=4)
            display = HTTPConnection("127.0.0.1", server.server_port, timeout=4)
            rows = []

            def poll():
                after = ""
                while not stop.is_set():
                    state = request(
                        display, "GET", "/state", {"X-Pair": "replay", "X-After": after, "X-Wait": "1"}
                    )
                    if state.get("image"):
                        after = str(state["seq"])
                        rows.append(
                            (
                                state["source_seq"],
                                state["seq"],
                                state["processing_ms"],
                                state["result_age_ms"],
                                bool(state.get("sim")),
                            )
                        )
                    elif not after:
                        stop.wait(0.01)

            reader = pool.submit(poll)
            try:
                headers = {"X-Pair": "12345678"}
                headers["X-Session"] = request(upload, "POST", "/camera/start", headers, b"")["session"]
                seq = request(upload, "POST", "/camera/ticket", headers, b"")["seq"]

                def send(data, media_time):
                    nonlocal seq
                    reply = request(
                        upload,
                        "POST",
                        "/camera/frame",
                        {**headers, "X-Seq": str(seq), "X-Media-Time": str(media_time), "X-Next-Ticket": "1"},
                        data,
                    )
                    seq = reply["next_seq"]

                warm = jpeg(frame)
                for i in range(30):
                    send(warm, i / fps)
                    time.sleep(0.025)
                if not rows:
                    raise RuntimeError(f"Worker produced no preview: {cell.snap.get('status')}")
                first = seq
                count = 0
                start = time.perf_counter()
                while ok:
                    data = jpeg(frame)
                    time.sleep(max(0, start + (count + 1) / fps - time.perf_counter()))
                    send(data, (count + 30) / fps)
                    count += 1
                    ok, frame = cap.read()
                elapsed = time.perf_counter() - start
                deadline = time.perf_counter() + 2
                while (not rows or rows[-1][0] != seq - 1) and time.perf_counter() < deadline:
                    time.sleep(0.01)
            finally:
                stop.set()
                upload.close()
                server.shutdown()
                try:
                    reader.result(timeout=5)
                finally:
                    display.close()
    measured = [r for r in rows if r[0] >= first]
    if not measured:
        raise RuntimeError("No frames received during measurement")
    print(
        f"Loopback input {count / elapsed:.2f} FPS; dashboard received {len(measured)}/{count} distinct frames, "
        f"{len(measured) / elapsed:.2f} FPS"
    )
    print(
        f"PC processing P95 {np.percentile([r[2] for r in measured], 95):.1f} ms; "
        f"ticket to result P95 {np.percentile([r[3] for r in measured], 95):.1f} ms"
    )
    print(f"Frames with a rendered 3D estimate: {sum(r[4] for r in measured)}/{len(measured)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, help="workcell JSON with camera intrinsics for this video")
    parser.add_argument(
        "--results", type=Path, help="optional per-frame JSON for reviewing estimated corners"
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto", help="cuda requires GPU inference"
    )
    parser.add_argument(
        "--trial-model",
        choices=("current", "detect", "pose", "pose-raw", "vit", "rescue"),
        default="pose",
        help="model to benchmark (default: verified YOLO11n-Pose + PnP; current: legacy CNN)",
    )
    parser.add_argument(
        "--stream-fps", type=float, help="replay through the live workcell on loopback at this FPS"
    )
    args = parser.parse_args()
    intr = json.loads(args.config.read_text(encoding="utf-8-sig")).get("intrinsics") if args.config else None
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        parser.error(f"Cannot open {args.video}")
    cv2.setNumThreads(2)
    if args.trial_model != "current":
        from twin_depot.cube_trial_observer import load_trial

        model = load_trial(args.trial_model, args.device)
        observer = model
    else:
        model = Completion(args.device)
        observer = FastObserver(model)
    print(f"Inference device: {model.device}")
    if args.stream_fps is not None:
        if not 0 < args.stream_fps <= 240 or args.results:
            parser.error("--stream-fps requires a rate in (0,240] and cannot be combined with --results")
        try:
            benchmark_stream(cap, args.config, model, args.stream_fps)
        finally:
            cap.release()
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    records = []
    timings = []
    i = 0
    while True:
        begin = time.perf_counter()
        ok, img = cap.read()
        if not ok:
            break
        start = time.perf_counter()
        k, dist = camera_matrix(img.shape, intr)
        _, partial, _ = observer.observe(img, None, k, dist, i / fps)
        end = time.perf_counter()
        timings.append([(end - start) * 1000, (end - begin) * 1000])
        if args.results:
            records.append(
                {
                    "i": i,
                    "ms": timings[-1][0],
                    "ob": [],
                    "part": partial,
                    "est": observer.estimates,
                    "pose_tracks": getattr(observer, "pose_tracks", []),
                }
            )
        i += 1
    cap.release()
    if not timings:
        parser.error("Video has no decodable frames")
    if args.results:
        args.results.write_text(json.dumps(records), encoding="utf8")
    warm = np.asarray(timings[1:] or timings)
    print(f"{i} frames, source {fps:.2f} FPS, first decode + pose {timings[0][1]:.2f} ms")
    for label, t in zip(("pose", "decode + pose"), warm.T, strict=True):
        median, p95 = np.percentile(t, [50, 95])
        print(
            f"{label}: median {median:.2f} ms, P95 {p95:.2f} ms, "
            f"max {t.max():.2f} ms, <=16.67 ms {np.mean(t <= 1000 / 60):.1%}, throughput {1000 / t.mean():.1f} FPS"
        )


if __name__ == "__main__":
    main()
