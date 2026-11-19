"""Latest-frame handoff for the phone uploader."""

import io
import math
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass

from PIL import Image

from .vision import FRESH_S


@dataclass(frozen=True)
class Frame:
    session: str
    seq: int = 0
    stamp: float = 0
    jpeg: bytes | None = None
    source_fps: float | None = None
    profile: dict | None = None

    @property
    def fresh(self):
        return self.seq > 0 and time.perf_counter() - self.stamp <= FRESH_S


class LatestFrames:
    def __init__(self):
        self.guard = threading.Lock()
        self.updated = threading.Condition(self.guard)
        self.start()

    def start(self, profile=None):
        with self.guard:
            self.frame = Frame(secrets.token_hex(10), profile=profile)
            self.seq = 0
            self.pending = self.media_time = None
            self.received = self.expired = self.duplicate = 0
            self.arrivals = deque(maxlen=40)
            self.updated.notify_all()
            return self.frame.session

    def ticket(self, session):
        with self.guard:
            if session != self.frame.session:
                raise ValueError("相机已重连，请重新开始")
            self.seq += 1
            self.pending = self.seq, time.perf_counter()
            return self.seq

    def receive(self, raw, session, seq, media_s=None):
        with self.guard:
            if session != self.frame.session or not self.pending or self.pending[0] != seq:
                raise ValueError("该帧已失效，请重新开始传输")
            _, stamp = self.pending
            self.pending = None
            self.received += 1
            if time.perf_counter() - stamp > FRESH_S:
                self.expired += 1
                return "expired"
            if media_s is not None:
                if not math.isfinite(media_s) or media_s < 0:
                    raise ValueError("无效的相机时间戳")
                if self.media_time is not None and media_s <= self.media_time:
                    self.duplicate += 1
                    return "duplicate"
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "JPEG" or max(image.size) > 2560 or min(image.size) < 64:
                    raise ValueError("需要 64 至 2560 像素的 JPEG 视频帧")
            self.media_time = media_s
            now = time.perf_counter()
            if self.arrivals and now - self.arrivals[-1] > FRESH_S:
                self.arrivals.clear()
            self.arrivals.append(now)
            fps = ((len(self.arrivals) - 1) / (now - self.arrivals[0])) if len(self.arrivals) > 1 else None
            self.frame = Frame(session, seq, stamp, raw, fps, self.frame.profile)
            self.updated.notify_all()

    def statistics(self):
        with self.guard:
            return dict(tickets=self.seq, received=self.received, expired=self.expired,
                        duplicate=self.duplicate, last_accepted_seq=self.frame.seq)

    def wait_next(self, after):
        with self.updated:
            self.updated.wait_for(lambda: (self.frame.session, self.frame.seq) != after, timeout=0.12)

    def read(self, after=None):
        with self.guard:
            return self.frame
