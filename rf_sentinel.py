#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-SENTINEL :: Real-Time Deep Learning RF Fingerprinting & Cyber-Security IDS
=============================================================================

Single-file reference implementation:

  * RF Generator thread  : 4 legitimate QPSK transmitters with distinct hardware
                           impairments (I/Q gain/phase imbalance, DC offset,
                           Saleh PA AM/AM + AM/PM saturation, oscillator
                           phase noise, residual CFO), an RF spoofer that clones
                           the carrier frequency AND the bit-exact payload of
                           the active device, and a high-power swept jammer.
  * DL backend (PyTorch) : 1D residual CNN (Conv1d + BatchNorm + skip links)
                           over raw 2x256 I/Q frames -> class probabilities
                           (Device 1-4 / Impostor / Jammer) + an L2-normalised
                           64-D physical-layer fingerprint embedding.
  * Security engine      : claimed-ID vs. predicted-ID check, cosine similarity
                           to the claimed device's fingerprint prototype,
                           RSSI anomaly detector, EMA smoothing + debouncing.
  * PyQt5/PyQtGraph GUI  : STFT waterfall + PSD, security console with
                           GREEN / RED / AMBER alert state and event log,
                           2-D PCA projection of the live latent space,
                           class-probability bars and fingerprint-similarity trace.

Model weights:
  The network is created with deterministic, seeded Kaiming/Xavier weights, so
  inference starts on the very first frame. Randomly initialised weights cannot
  classify anything, so on first launch a background bootstrap trainer
  (synthetic, physics-based data) refines the weights live and hot-swaps them
  into the running inference engine. The trained checkpoint is cached in
  ~/.cache/rf_sentinel/ so every later launch starts fully trained instantly.

Install (Ubuntu):
  pip install numpy torch pyqt5 pyqtgraph
  sudo apt install -y libxcb-xinerama0 libxkbcommon-x11-0   # Qt xcb runtime, if missing

Run:
  python3 rf_sentinel.py                  # auto CUDA / CPU
  python3 rf_sentinel.py --retrain        # ignore cached weights
  python3 rf_sentinel.py --cpu            # force CPU
  python3 rf_sentinel.py --train-steps 3000
"""

import os
import sys
import time
import queue
import argparse
import datetime
import threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# =============================================================================
# 1. SYSTEM CONSTANTS
# =============================================================================
FS_HZ = 2.0e6                 # complex sample rate of the simulated receiver
FRAME_LEN = 256               # I/Q samples per inference frame (2 x 256 tensor)
SPS = 8                       # samples per QPSK symbol
RRC_BETA = 0.35               # root-raised-cosine roll-off
RRC_SPAN = 4                  # RRC half-span in symbols
NSYM = FRAME_LEN // SPS + 2 * RRC_SPAN + 2
FPS = 30                      # frames per second produced by the generator
JSR_DB_RUNTIME = 18.0         # jammer-to-signal ratio of the live jammer
JAM_SWEEP_FRAMES = 45         # frames for one full sweep across the band
JAM_BASE_RATE = 0.9 / (JAM_SWEEP_FRAMES * FRAME_LEN)  # cycles/sample^2
WF_HISTORY = 240              # waterfall rows kept on screen
TRAIL_LEN = 220               # live points kept in the embedding scatter

NUM_DEVICES = 4
IMPOSTOR = 4
JAMMER = 5
NUM_CLASSES = 6
EMB_DIM = 64
CLASS_NAMES = ["Device 1", "Device 2", "Device 3", "Device 4", "Impostor", "Jammer"]
CLASS_SHORT = ["D1", "D2", "D3", "D4", "IMP", "JAM"]
CLASS_COLORS = ["#2ea8ff", "#3ddc84", "#b48cff", "#ffd166", "#ff4d4d", "#ff9f1c"]

# security decision parameters
EMA_ALPHA = 0.22              # probability smoothing
AUTH_PROB = 0.55              # min smoothed P(claimed device)
SIM_THRESHOLD = 0.55          # min cosine similarity to claimed fingerprint
RSSI_JAM_DB = 6.0             # RSSI excess (vs AGC reference) where jam score starts

WEIGHTS_VERSION = "rfnet-v1"
CKPT_PATH = os.path.join(os.path.expanduser("~"), ".cache", "rf_sentinel", WEIGHTS_VERSION + ".pt")

# Hardware fingerprint parameter vector layout
#   g     : I/Q gain imbalance (fraction)
#   phi   : I/Q quadrature phase skew (rad)
#   dci   : I-branch DC offset / LO leakage
#   dcq   : Q-branch DC offset / LO leakage
#   ba    : Saleh PA saturation coefficient (AM/AM & AM/PM denominator)
#   ap    : Saleh PA AM/PM conversion coefficient (rad)
#   drive : PA drive level (back-off)
#   pn    : oscillator phase-noise Wiener increment std (rad/sample)
#   cfo   : residual carrier frequency offset (cycles/sample)
PARAM_KEYS = ("g", "phi", "dci", "dcq", "ba", "ap", "drive", "pn", "cfo")
MULT_COLS = [0, 1, 4, 5, 6, 7]  # jittered multiplicatively (temperature / drift)
DC_COLS = [2, 3]                # jittered additively
CFO_COL = 8

LEGIT_PROFILES = np.array([
    #   g       phi(rad)               dci     dcq     ba    ap    drive  pn      cfo
    [0.020, np.deg2rad(2.0),          0.010, -0.005, 0.10, 0.10, 0.90, 0.0020,  2.0e-4],  # Device 1
    [0.075, np.deg2rad(-6.0),        -0.020,  0.015, 0.55, 0.25, 1.10, 0.0035, -3.0e-4],  # Device 2
    [-0.050, np.deg2rad(9.0),         0.000,  0.030, 0.20, 0.05, 0.70, 0.0100,  5.0e-4],  # Device 3
    [-0.010, np.deg2rad(-2.0),        0.045,  0.000, 0.30, 0.70, 1.00, 0.0050, -1.0e-4],  # Device 4
], dtype=np.float64)

# The live attacker: a commodity SDR with its own (different) analog front-end.
ROGUE_PROFILE = np.array(
    [0.110, np.deg2rad(13.0), 0.030, 0.030, 0.40, 0.40, 1.20, 0.0070, 0.0], dtype=np.float64)

IMP_LOW = np.array([-0.15, -0.26, -0.08, -0.08, 0.05, 0.0, 0.5, 0.001, -8e-4])
IMP_HIGH = np.array([0.15, 0.26, 0.08, 0.08, 0.70, 0.9, 1.3, 0.015, 8e-4])


# =============================================================================
# 2. PHYSICAL-LAYER SIGNAL MODEL (vectorised; shared by live generator & trainer)
# =============================================================================
def rrc_taps(beta, sps, span):
    """Unit-energy root-raised-cosine pulse."""
    t = np.arange(-span * sps, span * sps + 1, dtype=np.float64) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(4.0 * beta * ti) - 1.0) < 1e-9:
            h[i] = (beta / np.sqrt(2.0)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                                            (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


RRC = rrc_taps(RRC_BETA, SPS, RRC_SPAN)
_CONV_LEN = NSYM * SPS + len(RRC) - 1
RRC_NFFT = 1 << int(np.ceil(np.log2(_CONV_LEN)))
RRC_F = np.fft.fft(RRC, RRC_NFFT)
_RRC_DELAY = (len(RRC) - 1) // 2


def synthesize(params, snr_db, rng, payload=None, jam=None):
    """
    Generate a batch of received complex-baseband frames.

    params : dict PARAM_KEYS -> (n,) arrays (per-frame hardware fingerprint)
    snr_db : (n,) array
    payload: optional (n, NSYM) int array of QPSK symbol indices (0..3)
    jam    : optional dict with (n,) arrays: on, jsr_db, f0, rate, phase
    returns: (n, FRAME_LEN) complex64
    """
    snr_db = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    n = snr_db.shape[0]
    L = FRAME_LEN
    if payload is None:
        payload = rng.integers(0, 4, size=(n, NSYM))
    b0 = payload & 1
    b1 = (payload >> 1) & 1
    syms = ((1 - 2 * b0) + 1j * (1 - 2 * b1)) / np.sqrt(2.0)

    # pulse shaping (FFT convolution with RRC)
    up = np.zeros((n, NSYM * SPS), dtype=np.complex128)
    up[:, ::SPS] = syms
    shaped = np.fft.ifft(np.fft.fft(up, RRC_NFFT, axis=1) * RRC_F[None, :], axis=1)
    toff = rng.integers(0, SPS, size=n)  # random symbol timing phase
    start = _RRC_DELAY + RRC_SPAN * SPS + toff
    idx = start[:, None] + np.arange(L)[None, :]
    bb = np.take_along_axis(shaped, idx, axis=1)

    p = {k: np.asarray(params[k], dtype=np.float64).reshape(n, 1) for k in PARAM_KEYS}

    # I/Q modulator imbalance + LO leakage
    I, Q = bb.real, bb.imag
    i2 = (1.0 + p["g"]) * I + p["dci"]
    q2 = (1.0 - p["g"]) * (Q * np.cos(p["phi"]) - I * np.sin(p["phi"])) + p["dcq"]
    z = p["drive"] * (i2 + 1j * q2)

    # Saleh power amplifier (AM/AM compression + AM/PM conversion)
    r = np.abs(z)
    amp = r / (1.0 + p["ba"] * r ** 2)
    pm = p["ap"] * r ** 2 / (1.0 + p["ba"] * r ** 2)
    tx = amp * np.exp(1j * (np.angle(z) + pm))
    tx /= np.sqrt(np.mean(np.abs(tx) ** 2, axis=1, keepdims=True)) + 1e-12  # unit Tx power

    # oscillator: residual CFO + Wiener phase noise + random carrier phase
    t = np.arange(L, dtype=np.float64)[None, :]
    theta0 = rng.uniform(0.0, 2 * np.pi, size=(n, 1))
    phase_noise = np.cumsum(rng.standard_normal((n, L)) * p["pn"], axis=1)
    tx = tx * np.exp(1j * (2 * np.pi * p["cfo"] * t + phase_noise + theta0))

    # AWGN channel
    nvar = 10.0 ** (-snr_db.reshape(n, 1) / 10.0)
    rx = tx + np.sqrt(nvar / 2.0) * (rng.standard_normal((n, L)) + 1j * rng.standard_normal((n, L)))

    # swept (linear-chirp) jammer
    if jam is not None:
        on = np.asarray(jam["on"], dtype=np.float64).reshape(n, 1)
        if np.any(on):
            ja = np.sqrt(10.0 ** (np.asarray(jam["jsr_db"], dtype=np.float64).reshape(n, 1) / 10.0))
            f0 = np.asarray(jam["f0"], dtype=np.float64).reshape(n, 1)
            k = np.asarray(jam["rate"], dtype=np.float64).reshape(n, 1)
            ph0 = np.asarray(jam["phase"], dtype=np.float64).reshape(n, 1)
            rx = rx + on * ja * np.exp(1j * (2 * np.pi * (f0 * t + 0.5 * k * t ** 2) + ph0))
    return rx.astype(np.complex64)


def jitter_rows(rows, rng, scale=1.0):
    """Per-frame drift of the hardware fingerprint (temperature, supply, aging)."""
    out = np.array(rows, dtype=np.float64, copy=True)
    n = out.shape[0]
    out[:, MULT_COLS] *= 1.0 + rng.normal(0.0, 0.04 * scale, size=(n, len(MULT_COLS)))
    out[:, DC_COLS] += rng.normal(0.0, 0.003 * scale, size=(n, len(DC_COLS)))
    out[:, CFO_COL] += rng.normal(0.0, 2e-5 * scale, size=n)
    out[:, 4] = np.abs(out[:, 4]) + 1e-3   # ba > 0
    out[:, 7] = np.abs(out[:, 7])          # pn >= 0
    out[:, 6] = np.abs(out[:, 6]) + 1e-3   # drive > 0
    return out


def rows_to_params(rows):
    return {k: rows[:, i] for i, k in enumerate(PARAM_KEYS)}


def random_impostor_rows(n, rng):
    """Random rogue front-ends kept away from every authorised fingerprint."""
    span = IMP_HIGH - IMP_LOW
    out = rng.uniform(IMP_LOW, IMP_HIGH, size=(n, len(PARAM_KEYS)))
    for _ in range(12):
        d = np.linalg.norm((out[:, None, :8] - LEGIT_PROFILES[None, :, :8]) / span[:8], axis=2).min(axis=1)
        bad = d < 0.18
        if not np.any(bad):
            break
        out[bad] = rng.uniform(IMP_LOW, IMP_HIGH, size=(int(bad.sum()), len(PARAM_KEYS)))
    # an impostor clones the carrier: residual CFO equals a legit device's CFO
    out[:, CFO_COL] = LEGIT_PROFILES[rng.integers(0, NUM_DEVICES, n), CFO_COL]
    return out


def build_batch(labels, rng, snr_db, jsr_range=(8.0, 25.0)):
    """Synthesize frames for the given class labels (training / reference sets)."""
    labels = np.asarray(labels)
    n = labels.shape[0]
    rows = np.empty((n, len(PARAM_KEYS)), dtype=np.float64)

    legit = labels < NUM_DEVICES
    rows[legit] = LEGIT_PROFILES[labels[legit]]

    imp = labels == IMPOSTOR
    ni = int(imp.sum())
    if ni:
        r = random_impostor_rows(ni, rng)
        rogue = np.tile(ROGUE_PROFILE, (ni, 1))
        rogue[:, CFO_COL] = LEGIT_PROFILES[rng.integers(0, NUM_DEVICES, ni), CFO_COL]
        use_rogue = rng.random(ni) < 0.5
        r[use_rogue] = rogue[use_rogue]
        rows[imp] = r

    jm = labels == JAMMER
    nj = int(jm.sum())
    if nj:
        src = rng.integers(0, NUM_DEVICES + 1, nj)
        r = np.where((src < NUM_DEVICES)[:, None],
                     LEGIT_PROFILES[np.minimum(src, NUM_DEVICES - 1)],
                     np.tile(ROGUE_PROFILE, (nj, 1)))
        r[src == NUM_DEVICES, CFO_COL] = LEGIT_PROFILES[rng.integers(0, NUM_DEVICES, int((src == NUM_DEVICES).sum())), CFO_COL]
        rows[jm] = r

    rows = jitter_rows(rows, rng)
    jam = {
        "on": jm,
        "jsr_db": rng.uniform(jsr_range[0], jsr_range[1], n),
        "f0": rng.uniform(-0.45, 0.45, n),
        "rate": rng.uniform(0.3, 4.0, n) * JAM_BASE_RATE * rng.choice([-1.0, 1.0], n),
        "phase": rng.uniform(0, 2 * np.pi, n),
    }
    return synthesize(rows_to_params(rows), snr_db, rng, jam=jam)


def make_training_batch(n, rng, snr_range=(0.0, 30.0)):
    labels = rng.integers(0, NUM_CLASSES, n)
    snr = rng.uniform(snr_range[0], snr_range[1], n)
    return build_batch(labels, rng, snr), labels


def frames_to_tensor(rx):
    """(n, L) complex -> (n, 2, L) float32, power-normalised (receiver AGC)."""
    rx = np.asarray(rx)
    rms = np.sqrt(np.mean(np.abs(rx) ** 2, axis=1, keepdims=True)) + 1e-9
    x = rx / rms
    return np.stack([x.real, x.imag], axis=1).astype(np.float32)


def rssi_excess_db(rx):
    """Received power relative to the AGC reference level of one authorised carrier."""
    return 10.0 * np.log10(np.mean(np.abs(rx) ** 2) + 1e-12)


_WIN = np.hanning(FRAME_LEN).astype(np.float64)
_WIN_NORM = float(np.sum(_WIN ** 2))
FREQS_MHZ = np.fft.fftshift(np.fft.fftfreq(FRAME_LEN, 1.0 / FS_HZ)) / 1e6


def spectrum_db(x):
    X = np.fft.fftshift(np.fft.fft(x * _WIN))
    return (10.0 * np.log10(np.abs(X) ** 2 / _WIN_NORM + 1e-12)).astype(np.float32)


# =============================================================================
# 3. DEEP LEARNING BACKEND (PyTorch)
# =============================================================================
def select_device(force_cpu=False):
    if not force_cpu and torch.cuda.is_available():
        dev = torch.device("cuda:0")
        name = "CUDA :: " + torch.cuda.get_device_name(0)
    else:
        dev = torch.device("cpu")
        torch.set_num_threads(max(2, (os.cpu_count() or 2) - 1))
        name = "CPU :: %d threads%s" % (torch.get_num_threads(),
                                        "" if torch.cuda.is_available() or force_cpu else " (CUDA not available)")
    return dev, name


class ResidualBlock1D(nn.Module):
    """Conv1d-BN-ReLU-Conv1d-BN + identity / projection shortcut."""

    def __init__(self, cin, cout, stride=1, k=5):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                          nn.BatchNorm1d(cout))

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class RFFingerprintNet(nn.Module):
    """
    Input  : (B, 2, 256) raw I/Q
    Output : logits (B, 6), embedding (B, 64) L2-normalised PHY fingerprint
    """

    def __init__(self, in_ch=2, num_classes=NUM_CLASSES, emb_dim=EMB_DIM, width=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, 7, padding=3, bias=False),
                                  nn.BatchNorm1d(width), nn.ReLU(inplace=True))
        self.layer1 = ResidualBlock1D(width, width)
        self.layer2 = ResidualBlock1D(width, width * 2, stride=2)
        self.layer3 = ResidualBlock1D(width * 2, width * 4, stride=2)
        self.layer4 = ResidualBlock1D(width * 4, width * 4, stride=2)
        self.head = nn.Sequential(nn.Linear(width * 8, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.1), nn.Linear(128, emb_dim))
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.logit_scale = 12.0
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # zero-init last BN of each residual branch -> blocks start as identity
        for m in self.modules():
            if isinstance(m, ResidualBlock1D):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        h = self.stem(x)
        h = self.layer4(self.layer3(self.layer2(self.layer1(h))))
        h = torch.cat([F.adaptive_avg_pool1d(h, 1), F.adaptive_max_pool1d(h, 1)], dim=1).flatten(1)
        emb = F.normalize(self.head(h), dim=1)
        logits = self.classifier(emb) * self.logit_scale
        return logits, emb


class ModelHub:
    """Thread-safe holder of the live inference model (hot-swappable weights)."""

    def __init__(self, device):
        self.device = device
        self.lock = threading.Lock()
        torch.manual_seed(1337)  # deterministic pre-initialised weights
        self.model = RFFingerprintNet().to(device).eval()
        self.version = 0
        self.ready = False
        self.val_acc = None

    def publish(self, state_dict, ready=False, val_acc=None):
        with self.lock:
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.version += 1
            self.ready = self.ready or ready
            if val_acc is not None:
                self.val_acc = val_acc

    def state_dict_cpu(self):
        with self.lock:
            return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    @torch.no_grad()
    def infer(self, x_np):
        with self.lock:
            x = torch.from_numpy(x_np).to(self.device)
            logits, emb = self.model(x)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy(), emb.cpu().numpy(), self.version


def load_checkpoint(hub, path=CKPT_PATH):
    if not os.path.isfile(path):
        return False
    try:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            ck = torch.load(path, map_location="cpu")
        if ck.get("version") != WEIGHTS_VERSION:
            return False
        hub.publish(ck["state_dict"], ready=True, val_acc=ck.get("val_acc"))
        return True
    except Exception as exc:  # corrupted / incompatible cache -> retrain
        print("[RF-SENTINEL] checkpoint ignored:", exc)
        return False


@torch.no_grad()
def evaluate_model(model, device, rng, n_per_class=200, snr_db=20.0):
    model.eval()
    labels = np.repeat(np.arange(NUM_CLASSES), n_per_class)
    rx = build_batch(labels, rng, np.full(labels.shape[0], snr_db))
    x = torch.from_numpy(frames_to_tensor(rx)).to(device)
    logits, _ = model(x)
    pred = logits.argmax(1).cpu().numpy()
    per_class = [float(np.mean(pred[labels == c] == c)) for c in range(NUM_CLASSES)]
    return float(np.mean(pred == labels)), per_class


class TrainerThread(QtCore.QThread):
    """Bootstraps / refines the fingerprint CNN and hot-swaps weights into the hub."""
    progress = QtCore.pyqtSignal(object)

    def __init__(self, hub, steps, batch, ckpt_path=CKPT_PATH):
        super().__init__()
        self.hub, self.steps, self.batch, self.ckpt_path = hub, steps, batch, ckpt_path
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self._train()
        except Exception as exc:
            self.progress.emit({"error": str(exc)})

    def _train(self):
        dev = self.hub.device
        model = RFFingerprintNet().to(dev)
        model.load_state_dict(self.hub.state_dict_cpu())
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=self.steps, pct_start=0.15)
        rng = np.random.default_rng(2024)
        publish_every = max(20, self.steps // 40)
        loss_ema, acc_ema = None, None
        t0 = time.perf_counter()
        for step in range(1, self.steps + 1):
            if self._stop:
                return
            rx, y = make_training_batch(self.batch, rng)
            x = torch.from_numpy(frames_to_tensor(rx)).to(dev)
            yt = torch.from_numpy(y).long().to(dev)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, yt, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            acc = (logits.argmax(1) == yt).float().mean().item()
            lv = loss.item()
            loss_ema = lv if loss_ema is None else 0.9 * loss_ema + 0.1 * lv
            acc_ema = acc if acc_ema is None else 0.9 * acc_ema + 0.1 * acc
            if step % publish_every == 0 or step == self.steps:
                model.eval()
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                model.train()
                self.hub.publish(sd, ready=(step >= 0.3 * self.steps))
                self.progress.emit({"step": step, "steps": self.steps, "loss": loss_ema,
                                    "acc": acc_ema, "elapsed": time.perf_counter() - t0})
        val_acc, per_class = evaluate_model(model, dev, np.random.default_rng(99))
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.hub.publish(sd, ready=True, val_acc=val_acc)
        try:
            os.makedirs(os.path.dirname(self.ckpt_path), exist_ok=True)
            torch.save({"version": WEIGHTS_VERSION, "state_dict": sd, "val_acc": val_acc,
                        "per_class": per_class}, self.ckpt_path)
            saved = self.ckpt_path
        except Exception as exc:
            saved = "not saved (%s)" % exc
        self.progress.emit({"done": True, "val_acc": val_acc, "per_class": per_class,
                            "elapsed": time.perf_counter() - t0, "ckpt": saved})


# =============================================================================
# 4. REAL-TIME THREADS: RF GENERATOR + INFERENCE ENGINE
# =============================================================================
class RFGeneratorThread(QtCore.QThread):
    """Synthetic SDR front-end producing one 256-sample I/Q frame every 1/FPS s."""

    def __init__(self, out_queue):
        super().__init__()
        self.q = out_queue
        self.active = 0
        self.spoof = False
        self.jam = False
        self.snr_db = 18.0
        self._running = True
        self.rng = np.random.default_rng()
        self.frame_id = 0
        self.j_f0 = -0.45
        self.j_phase = 0.0

    def stop(self):
        self._running = False

    def _payload(self, dev):
        # deterministic per (device, frame): the spoofer replays it bit-exactly
        prng = np.random.default_rng((dev + 1) * 1_000_003 + self.frame_id)
        return prng.integers(0, 4, size=(1, NSYM))

    def run(self):
        period = 1.0 / FPS
        nxt = time.perf_counter()
        while self._running:
            dev, spoof, jam, snr = self.active, self.spoof, self.jam, self.snr_db
            if spoof:
                base = ROGUE_PROFILE.copy()
                base[CFO_COL] = LEGIT_PROFILES[dev, CFO_COL]  # carrier-frequency cloning
            else:
                base = LEGIT_PROFILES[dev].copy()
            rows = jitter_rows(base[None, :], self.rng)
            jam_d = None
            if jam:
                jam_d = {"on": np.array([True]), "jsr_db": np.array([JSR_DB_RUNTIME]),
                         "f0": np.array([self.j_f0]), "rate": np.array([JAM_BASE_RATE]),
                         "phase": np.array([self.j_phase])}
                L = FRAME_LEN
                self.j_phase = (self.j_phase + 2 * np.pi * (self.j_f0 * L + 0.5 * JAM_BASE_RATE * L * L)) % (2 * np.pi)
                self.j_f0 += JAM_BASE_RATE * L
                if self.j_f0 > 0.45:
                    self.j_f0 = -0.45
            rx = synthesize(rows_to_params(rows), np.array([snr]), self.rng,
                            payload=self._payload(dev), jam=jam_d)[0]
            truth = JAMMER if jam else (IMPOSTOR if spoof else dev)
            meta = {"frame": self.frame_id, "claimed": dev, "truth": truth, "spoof": spoof,
                    "jam": jam, "snr": snr, "t": time.time()}
            try:
                self.q.put_nowait((rx, meta))
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait((rx, meta))
                except queue.Full:
                    pass
            self.frame_id += 1
            nxt += period
            dt = nxt - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                nxt = time.perf_counter()


class InferenceThread(QtCore.QThread):
    """Batches incoming frames, runs the CNN, projects embeddings, emits results."""
    results = QtCore.pyqtSignal(object)
    reference = QtCore.pyqtSignal(object)

    def __init__(self, in_queue, hub):
        super().__init__()
        self.q, self.hub = in_queue, hub
        self._running = True
        self.ref_version = -1
        self.ref_time = 0.0
        self.mean = None
        self.basis = None
        self.protos = None

    def stop(self):
        self._running = False

    def _build_reference(self):
        """Reference fingerprints -> class prototypes + fixed 2-D PCA basis."""
        rng = np.random.default_rng(7)
        per = 60
        labels = np.repeat(np.arange(NUM_CLASSES), per)
        rx = build_batch(labels, rng, np.full(labels.shape[0], 20.0), jsr_range=(JSR_DB_RUNTIME, JSR_DB_RUNTIME))
        _, emb, ver = self.hub.infer(frames_to_tensor(rx))
        protos = np.stack([emb[labels == c].mean(0) for c in range(NUM_CLASSES)])
        protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9
        mean = emb.mean(0)
        _, _, vt = np.linalg.svd(emb - mean, full_matrices=False)
        basis = vt[:2].copy()
        if self.basis is not None:  # keep orientation stable between weight updates
            for i in range(2):
                if np.dot(basis[i], self.basis[i]) < 0:
                    basis[i] = -basis[i]
        self.mean, self.basis, self.protos = mean, basis, protos
        self.ref_version = ver
        self.ref_time = time.perf_counter()
        self.reference.emit({"points": (emb - mean) @ basis.T, "labels": labels,
                             "protos2d": (protos - mean) @ basis.T, "version": ver})

    def run(self):
        while self._running:
            try:
                first = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            items = [first]
            while len(items) < 16:
                try:
                    items.append(self.q.get_nowait())
                except queue.Empty:
                    break
            if self.hub.version != self.ref_version and (time.perf_counter() - self.ref_time > 1.5 or self.basis is None):
                self._build_reference()
            rx = np.stack([it[0] for it in items])
            t0 = time.perf_counter()
            probs, emb, _ = self.hub.infer(frames_to_tensor(rx))
            latency = (time.perf_counter() - t0) * 1000.0
            proj = (emb - self.mean) @ self.basis.T
            sims = emb @ self.protos.T
            out = []
            for i, (frame, meta) in enumerate(items):
                out.append({"rx": frame, "meta": meta, "probs": probs[i], "emb2d": proj[i],
                            "sims": sims[i], "rssi_db": rssi_excess_db(frame),
                            "latency_ms": latency, "batch": len(items)})
            self.results.emit(out)


# =============================================================================
# 5. SECURITY DECISION ENGINE
# =============================================================================
class SecurityEngine:
    """Fuses classifier, fingerprint similarity and RSSI into a debounced alert state."""

    def __init__(self):
        self.reset()
        self.state = "CALIBRATING"

    def reset(self):
        self.ema = None
        self.sim_ema = None
        self.cand = None
        self.cand_n = 0
        self.grace = 8  # frames to let the EMA re-converge after an operator change

    def evaluate(self, probs, sims, claimed, rssi_db, ready):
        self.ema = probs.copy() if self.ema is None else EMA_ALPHA * probs + (1 - EMA_ALPHA) * self.ema
        sim = float(sims[claimed])
        self.sim_ema = sim if self.sim_ema is None else EMA_ALPHA * sim + (1 - EMA_ALPHA) * self.sim_ema
        p = self.ema
        pred = int(np.argmax(p))
        rssi_score = float(np.clip((rssi_db - RSSI_JAM_DB) / 6.0, 0.0, 1.0))
        jam_score = max(float(p[JAMMER]), rssi_score)
        if not ready:
            raw = "CALIBRATING"
        elif jam_score >= 0.5:
            raw = "JAM"
        elif pred == claimed and p[claimed] >= AUTH_PROB and self.sim_ema >= SIM_THRESHOLD:
            raw = "AUTH"
        else:
            raw = "SPOOF"
        if self.grace > 0:
            self.grace -= 1
            if raw == "SPOOF" and self.state != "CALIBRATING":
                raw = self.state  # hold the current state while the smoother settles

        changed, prev = False, self.state
        if raw == self.state:
            self.cand, self.cand_n = None, 0
        else:
            if raw == self.cand:
                self.cand_n += 1
            else:
                self.cand, self.cand_n = raw, 1
            need = {"JAM": 2, "SPOOF": 5}.get(raw, 4)  # ~0.07-0.17 s at 30 fps; debounces false alarms
            if self.state == "CALIBRATING" or self.cand_n >= need:
                self.state = raw
                self.cand, self.cand_n = None, 0
                changed = True
        return {"state": self.state, "raw": raw, "changed": changed, "prev": prev, "pred": pred,
                "p": p, "sim": self.sim_ema, "jam_score": jam_score, "rssi_score": rssi_score}


# =============================================================================
# 6. GUI
# =============================================================================
STATE_STYLE = {
    "AUTH": ("#0f5132", "#3ddc84", "\u25CF  AUTHENTICATED TRANSMISSION"),
    "SPOOF": ("#5c0b12", "#ff4d4d", "\u25B2  RF SPOOFING / IMPOSTOR DETECTED"),
    "JAM": ("#5a3a00", "#ffb000", "\u25C6  ACTIVE JAMMING ATTACK"),
    "CALIBRATING": ("#1c2a3a", "#58a6ff", "\u25CC  CALIBRATING DEEP FINGERPRINT MODEL"),
}
EXPECTED_STATE = {0: "AUTH", 1: "AUTH", 2: "AUTH", 3: "AUTH", IMPOSTOR: "SPOOF", JAMMER: "JAM"}

APP_STYLE = """
QWidget { background-color: #0b0f14; color: #c9d1d9; font-family: 'DejaVu Sans', 'Ubuntu', sans-serif; font-size: 10pt; }
QGroupBox { border: 1px solid #263241; border-radius: 6px; margin-top: 16px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #58a6ff; font-weight: bold; }
QComboBox, QPushButton { background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 6px 12px; }
QComboBox QAbstractItemView { background-color: #161b22; selection-background-color: #1f6feb; }
QPushButton:hover { border-color: #58a6ff; }
QPushButton#spoofBtn:checked { background-color: #8b1a1a; border-color: #ff4d4d; color: #ffffff; font-weight: bold; }
QPushButton#jamBtn:checked { background-color: #7a4f00; border-color: #ffb000; color: #ffffff; font-weight: bold; }
QPlainTextEdit { background-color: #05080b; border: 1px solid #263241; font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; }
QSlider::groove:horizontal { height: 6px; background: #30363d; border-radius: 3px; }
QSlider::handle:horizontal { background: #58a6ff; width: 14px; margin: -5px 0; border-radius: 7px; }
QLabel#metrics { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; }
"""


def make_lut():
    pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cols = np.array([[0, 0, 4, 255], [40, 11, 84, 255], [137, 34, 106, 255],
                     [229, 92, 48, 255], [252, 255, 164, 255]], dtype=np.ubyte)
    return pg.ColorMap(pos, cols).getLookupTable(0.0, 1.0, 256)


def place_image(img, x0, y0, w, h, nx, ny):
    tr = QtGui.QTransform()
    tr.translate(x0, y0)
    tr.scale(w / nx, h / ny)
    img.setTransform(tr)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, hub, device_name, train_steps, train_batch, weights_loaded):
        super().__init__()
        self.hub = hub
        self.device_name = device_name
        self.setWindowTitle("RF-SENTINEL  |  Deep RF Fingerprinting & PHY-Layer Intrusion Detection")
        self.resize(1680, 980)

        self.frame_q = queue.Queue(maxsize=48)
        self.gen = RFGeneratorThread(self.frame_q)
        self.inf = InferenceThread(self.frame_q, hub)
        self.engine = SecurityEngine()
        self.trainer = None

        self.wf_data = np.full((WF_HISTORY, FRAME_LEN), -35.0, dtype=np.float32)
        self.psd_avg = None
        self.trail = deque(maxlen=TRAIL_LEN)
        self.sim_hist = deque(maxlen=300)
        self.correct_hist = deque(maxlen=300)
        self.frames = 0
        self.alerts = 0
        self.model_text = ""
        self.train_pct = 0
        self.last_rate_t = time.perf_counter()
        self.last_rate_frames = 0
        self.fps = 0.0
        self.brushes = [pg.mkBrush(QtGui.QColor(c)) for c in CLASS_COLORS]
        self.brushes_ref = []
        for c in CLASS_COLORS:
            qc = QtGui.QColor(c)
            qc.setAlpha(45)
            self.brushes_ref.append(pg.mkBrush(qc))

        self._build_ui()
        self.inf.results.connect(self.on_results)
        self.inf.reference.connect(self.on_reference)

        self.log("SYSTEM", "RF-SENTINEL online. Compute backend: %s" % device_name, "#58a6ff")
        if weights_loaded:
            acc = hub.val_acc
            self.model_text = "Model: cached weights loaded" + (" (val acc %.1f%% @20 dB)" % (100 * acc) if acc else "")
            self.log("MODEL", "Loaded trained fingerprint weights from %s" % CKPT_PATH, "#58a6ff")
        else:
            self.model_text = "Model: seeded weights, bootstrap training started"
            self.log("MODEL", "No cached weights - inference running on seeded weights; "
                              "bootstrap trainer refining live (%d steps)." % train_steps, "#58a6ff")
            self.trainer = TrainerThread(hub, train_steps, train_batch)
            self.trainer.progress.connect(self.on_train_progress)
            self.trainer.start()
        self.model_lbl.setText(self.model_text)

        self.gen.start()
        self.inf.start()

    # ---------------------------------------------------------------- UI build
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # ---- control bar
        bar = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("RF-SENTINEL")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; color: #58a6ff; letter-spacing: 2px;")
        bar.addWidget(title)
        bar.addSpacing(18)
        bar.addWidget(QtWidgets.QLabel("Active Transmitter:"))
        self.dev_combo = QtWidgets.QComboBox()
        self.dev_combo.addItems(["Device %d  (authorised)" % (i + 1) for i in range(NUM_DEVICES)])
        self.dev_combo.currentIndexChanged.connect(self.on_device_changed)
        bar.addWidget(self.dev_combo)
        bar.addSpacing(10)
        self.spoof_btn = QtWidgets.QPushButton("Launch RF Spoofing Attack")
        self.spoof_btn.setObjectName("spoofBtn")
        self.spoof_btn.setCheckable(True)
        self.spoof_btn.toggled.connect(self.on_spoof_toggled)
        bar.addWidget(self.spoof_btn)
        self.jam_btn = QtWidgets.QPushButton("Launch Swept Jammer")
        self.jam_btn.setObjectName("jamBtn")
        self.jam_btn.setCheckable(True)
        self.jam_btn.toggled.connect(self.on_jam_toggled)
        bar.addWidget(self.jam_btn)
        bar.addSpacing(14)
        bar.addWidget(QtWidgets.QLabel("SNR:"))
        self.snr_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.snr_slider.setRange(0, 30)
        self.snr_slider.setValue(18)
        self.snr_slider.setFixedWidth(200)
        self.snr_slider.valueChanged.connect(self.on_snr_changed)
        bar.addWidget(self.snr_slider)
        self.snr_lbl = QtWidgets.QLabel("18 dB")
        self.snr_lbl.setFixedWidth(48)
        bar.addWidget(self.snr_lbl)
        bar.addStretch(1)
        dev_lbl = QtWidgets.QLabel(self.device_name)
        dev_lbl.setStyleSheet("color: %s; font-weight: bold;" % ("#3ddc84" if "CUDA" in self.device_name else "#ffd166"))
        bar.addWidget(dev_lbl)
        root.addLayout(bar)

        self.model_lbl = QtWidgets.QLabel("")
        self.model_lbl.setStyleSheet("color: #8b949e;")
        root.addWidget(self.model_lbl)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        root.addLayout(grid, 1)

        # ---- Display 1: waterfall + PSD
        g1 = QtWidgets.QGroupBox("Display 1 :: Live RF Waterfall Spectrogram (STFT, 256-pt Hann)")
        l1 = QtWidgets.QVBoxLayout(g1)
        glw = pg.GraphicsLayoutWidget()
        l1.addWidget(glw)
        self.psd_plot = glw.addPlot(row=0, col=0)
        self.psd_plot.setLabel("left", "PSD", units="dB")
        self.psd_plot.setYRange(-40, 50)
        self.psd_plot.setXRange(FREQS_MHZ[0], FREQS_MHZ[-1], padding=0)
        self.psd_plot.showGrid(x=True, y=True, alpha=0.25)
        self.psd_plot.setMaximumHeight(170)
        self.psd_curve = self.psd_plot.plot(FREQS_MHZ, np.full(FRAME_LEN, -35.0), pen=pg.mkPen("#58a6ff", width=1.5))
        self.psd_inst = self.psd_plot.plot(FREQS_MHZ, np.full(FRAME_LEN, -35.0), pen=pg.mkPen((88, 166, 255, 70), width=1))
        self.wf_plot = glw.addPlot(row=1, col=0)
        self.wf_plot.setLabel("bottom", "Frequency offset", units="MHz")
        self.wf_plot.setLabel("left", "Time (s, newest on top)")
        self.wf_img = pg.ImageItem()
        self.wf_img.setLookupTable(make_lut())
        self.wf_img.setImage(self.wf_data, autoLevels=False, levels=(-35, 40))
        hist_s = WF_HISTORY / float(FPS)
        place_image(self.wf_img, FREQS_MHZ[0], -hist_s, FS_HZ / 1e6, hist_s, FRAME_LEN, WF_HISTORY)
        self.wf_plot.addItem(self.wf_img)
        self.wf_plot.setXRange(FREQS_MHZ[0], FREQS_MHZ[-1], padding=0)
        self.wf_plot.setYRange(-hist_s, 0, padding=0)
        self.psd_plot.setXLink(self.wf_plot)
        grid.addWidget(g1, 0, 0)

        # ---- Display 2: security console
        g2 = QtWidgets.QGroupBox("Display 2 :: Real-Time Security Console")
        l2 = QtWidgets.QVBoxLayout(g2)
        self.status_lbl = QtWidgets.QLabel()
        self.status_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.status_lbl.setMinimumHeight(74)
        self.status_lbl.setWordWrap(True)
        f = QtGui.QFont()
        f.setPointSize(15)
        f.setBold(True)
        self.status_lbl.setFont(f)
        l2.addWidget(self.status_lbl)
        self.metrics_lbl = QtWidgets.QLabel()
        self.metrics_lbl.setObjectName("metrics")
        self.metrics_lbl.setTextFormat(QtCore.Qt.RichText)
        l2.addWidget(self.metrics_lbl)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(600)
        l2.addWidget(self.log_box, 1)
        grid.addWidget(g2, 0, 1)
        self.set_status("CALIBRATING", claimed=0)

        # ---- Display 3: embedding scatter
        g3 = QtWidgets.QGroupBox("Display 3 :: PHY Fingerprint Latent Space (64-D -> 2-D PCA)")
        l3 = QtWidgets.QVBoxLayout(g3)
        self.emb_plot = pg.PlotWidget()
        l3.addWidget(self.emb_plot)
        self.emb_plot.showGrid(x=True, y=True, alpha=0.2)
        self.emb_plot.setLabel("bottom", "PC 1")
        self.emb_plot.setLabel("left", "PC 2")
        self.emb_plot.setAspectLocked(False)
        for ax in ("bottom", "left"):
            self.emb_plot.getAxis(ax).enableAutoSIPrefix(False)
        legend = self.emb_plot.addLegend(offset=(8, 8))
        for c in range(NUM_CLASSES):  # legend entries
            dummy = pg.ScatterPlotItem([], [], brush=self.brushes[c], pen=None, size=9, name=CLASS_NAMES[c])
            self.emb_plot.addItem(dummy)
        self.ref_scatter = pg.ScatterPlotItem(pen=None, size=6)
        self.trail_scatter = pg.ScatterPlotItem(pen=None, size=7)
        self.live_marker = pg.ScatterPlotItem(pen=pg.mkPen("#ffffff", width=2), size=17, symbol="o")
        self.proto_scatter = pg.ScatterPlotItem(pen=pg.mkPen("#ffffff", width=1.5), size=15, symbol="x")
        for it in (self.ref_scatter, self.trail_scatter, self.proto_scatter, self.live_marker):
            self.emb_plot.addItem(it)
        self.proto_labels = []
        for c in range(NUM_CLASSES):
            ti = pg.TextItem(CLASS_SHORT[c], color=CLASS_COLORS[c], anchor=(0.5, 1.4))
            self.emb_plot.addItem(ti)
            self.proto_labels.append(ti)
        grid.addWidget(g3, 1, 0)

        # ---- analytics: class probabilities + similarity trace
        g4 = QtWidgets.QGroupBox("Neural Classifier Output  ::  Fingerprint Similarity to Claimed Identity")
        l4 = QtWidgets.QVBoxLayout(g4)
        glw2 = pg.GraphicsLayoutWidget()
        l4.addWidget(glw2)
        self.prob_plot = glw2.addPlot(row=0, col=0)
        self.prob_plot.setYRange(0, 1.05, padding=0)
        self.prob_plot.setXRange(-0.6, NUM_CLASSES - 0.4, padding=0)
        self.prob_plot.setLabel("left", "P(class)")
        self.prob_plot.getAxis("bottom").setTicks([[(i, CLASS_NAMES[i]) for i in range(NUM_CLASSES)]])
        self.prob_plot.showGrid(y=True, alpha=0.2)
        self.prob_bars = pg.BarGraphItem(x=np.arange(NUM_CLASSES), height=np.zeros(NUM_CLASSES), width=0.62,
                                         brushes=self.brushes, pens=[pg.mkPen(None)] * NUM_CLASSES)
        self.prob_plot.addItem(self.prob_bars)
        self.sim_plot = glw2.addPlot(row=1, col=0)
        self.sim_plot.setYRange(-0.2, 1.05, padding=0)
        self.sim_plot.setLabel("left", "cos-sim")
        self.sim_plot.setLabel("bottom", "frames (last 10 s)")
        self.sim_plot.showGrid(x=True, y=True, alpha=0.2)
        self.sim_curve = self.sim_plot.plot(pen=pg.mkPen("#3ddc84", width=2))
        self.sim_plot.addItem(pg.InfiniteLine(pos=SIM_THRESHOLD, angle=0,
                                              pen=pg.mkPen("#ff4d4d", width=1, style=QtCore.Qt.DashLine)))
        grid.addWidget(g4, 1, 1)

        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

    # ---------------------------------------------------------------- helpers
    def log(self, tag, msg, color="#c9d1d9"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.appendHtml('<span style="color:#6e7681">[%s]</span> <span style="color:%s"><b>%-7s</b> %s</span>'
                                % (ts, color, tag, msg))

    def set_status(self, state, claimed, extra=""):
        bg, fg, text = STATE_STYLE[state]
        if state == "AUTH":
            text += "\nDevice %d  |  PHY fingerprint verified" % (claimed + 1)
        elif state == "SPOOF":
            text += "\nTransmitter claims Device %d  |  fingerprint MISMATCH" % (claimed + 1)
        elif state == "JAM":
            text += "\nHigh-power swept interference  |  link integrity compromised"
        elif state == "CALIBRATING":
            text += "\n%s" % (extra or "bootstrapping weights")
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet("background-color:%s; color:%s; border:2px solid %s; border-radius:8px; padding:6px;"
                                      % (bg, fg, fg))

    # ---------------------------------------------------------------- controls
    def on_device_changed(self, idx):
        self.gen.active = idx
        self.engine.reset()
        self.log("OPERATOR", "Active transmitter switched to Device %d" % (idx + 1), "#8b949e")

    def on_spoof_toggled(self, on):
        self.gen.spoof = on
        self.engine.reset()
        self.spoof_btn.setText("STOP RF Spoofing Attack" if on else "Launch RF Spoofing Attack")
        if on:
            self.log("ATTACK", "Spoofer locked on Device %d carrier; replaying bit-exact payload with rogue SDR front-end"
                     % (self.gen.active + 1), "#ff7b72")
        else:
            self.log("ATTACK", "Spoofing attack stopped - legitimate Device %d back on air" % (self.gen.active + 1), "#8b949e")

    def on_jam_toggled(self, on):
        self.gen.jam = on
        self.engine.reset()
        self.jam_btn.setText("STOP Swept Jammer" if on else "Launch Swept Jammer")
        if on:
            self.log("ATTACK", "Swept jammer active: linear chirp across +/-0.9 MHz, J/S = %+.0f dB" % JSR_DB_RUNTIME, "#ffb000")
        else:
            self.log("ATTACK", "Swept jammer stopped", "#8b949e")

    def on_snr_changed(self, v):
        self.gen.snr_db = float(v)
        self.snr_lbl.setText("%d dB" % v)

    # ---------------------------------------------------------------- training callbacks
    def on_train_progress(self, info):
        if "error" in info:
            self.log("MODEL", "Trainer error: %s" % info["error"], "#ff4d4d")
            return
        if info.get("done"):
            pc = " ".join("%s:%.0f%%" % (CLASS_SHORT[i], 100 * a) for i, a in enumerate(info["per_class"]))
            self.model_text = "Model: trained in %.0f s  |  val acc %.1f%% @20 dB  [%s]" % (
                info["elapsed"], 100 * info["val_acc"], pc)
            self.log("MODEL", "Bootstrap complete - validation accuracy %.1f%% @ 20 dB. Weights cached at %s"
                     % (100 * info["val_acc"], info["ckpt"]), "#3ddc84")
            self.train_pct = 100
        else:
            self.train_pct = int(100 * info["step"] / info["steps"])
            self.model_text = "Model: bootstrap training %d%%  |  step %d/%d  |  loss %.3f  |  train acc %.1f%%  (weights hot-swapped live)" % (
                self.train_pct, info["step"], info["steps"], info["loss"], 100 * info["acc"])
        self.model_lbl.setText(self.model_text)

    def on_reference(self, ref):
        pts, labels = ref["points"], ref["labels"]
        self.ref_scatter.setData(x=pts[:, 0], y=pts[:, 1], brush=[self.brushes_ref[c] for c in labels])
        pr = ref["protos2d"]
        self.proto_scatter.setData(x=pr[:, 0], y=pr[:, 1], brush=self.brushes)
        for c in range(NUM_CLASSES):
            self.proto_labels[c].setPos(float(pr[c, 0]), float(pr[c, 1]))
        self.trail.clear()  # projection basis changed -> drop stale live points

    # ---------------------------------------------------------------- live results
    def on_results(self, batch):
        last = None
        for r in batch:
            row = spectrum_db(r["rx"])
            self.wf_data[:-1] = self.wf_data[1:]
            self.wf_data[-1] = row  # newest row (rendered on top)
            self.psd_avg = row if self.psd_avg is None else 0.25 * row + 0.75 * self.psd_avg
            meta = r["meta"]
            res = self.engine.evaluate(r["probs"], r["sims"], meta["claimed"], r["rssi_db"], self.hub.ready)
            self.trail.append((float(r["emb2d"][0]), float(r["emb2d"][1]), int(np.argmax(r["probs"]))))
            self.sim_hist.append(float(r["sims"][meta["claimed"]]))
            if self.hub.ready:
                self.correct_hist.append(res["raw"] == EXPECTED_STATE[meta["truth"]])
            if res["changed"]:
                self.on_state_change(res, meta)
            self.frames += 1
            last = (r, res, row)

        if last is None:
            return
        r, res, row = last
        meta = r["meta"]
        self.wf_img.setImage(self.wf_data, autoLevels=False, levels=(-35, 40))
        self.psd_curve.setData(FREQS_MHZ, self.psd_avg)
        self.psd_inst.setData(FREQS_MHZ, row)

        if self.trail:
            tr = np.array(self.trail)
            cls = tr[:, 2].astype(int)
            self.trail_scatter.setData(x=tr[:, 0], y=tr[:, 1], brush=[self.brushes[c] for c in cls])
            self.live_marker.setData(x=[tr[-1, 0]], y=[tr[-1, 1]], brush=[self.brushes[cls[-1]]])

        self.prob_bars.setOpts(height=res["p"])
        self.sim_curve.setData(np.arange(len(self.sim_hist)), np.array(self.sim_hist))
        self.sim_curve.setPen(pg.mkPen("#3ddc84" if res["sim"] >= SIM_THRESHOLD else "#ff4d4d", width=2))

        if res["state"] == "CALIBRATING":
            self.set_status("CALIBRATING", meta["claimed"], "model warm-up %d%%" % self.train_pct)

        now = time.perf_counter()
        if now - self.last_rate_t >= 1.0:
            self.fps = (self.frames - self.last_rate_frames) / (now - self.last_rate_t)
            self.last_rate_t, self.last_rate_frames = now, self.frames

        acc = (100.0 * np.mean(self.correct_hist)) if self.correct_hist else float("nan")
        pred = res["pred"]
        truth = CLASS_NAMES[meta["truth"]] if meta["truth"] < NUM_DEVICES else (
            "Spoofer (rogue SDR)" if meta["truth"] == IMPOSTOR else "Swept jammer")
        self.metrics_lbl.setText(
            "<table cellspacing='3'>"
            "<tr><td>Claimed identity</td><td><b>Device %d</b></td><td>&nbsp;&nbsp;Predicted</td>"
            "<td><b style='color:%s'>%s</b> (%.1f%%)</td></tr>"
            "<tr><td>Fingerprint cos-sim</td><td><b style='color:%s'>%.3f</b> (thr %.2f)</td>"
            "<td>&nbsp;&nbsp;RSSI excess</td><td><b>%+.1f dB</b></td></tr>"
            "<tr><td>Jam score</td><td><b>%.2f</b></td><td>&nbsp;&nbsp;Inference</td>"
            "<td><b>%.2f ms</b> / batch of %d</td></tr>"
            "<tr><td>Throughput</td><td><b>%.1f fps</b></td><td>&nbsp;&nbsp;Frames / alerts</td>"
            "<td><b>%d</b> / <b>%d</b></td></tr>"
            "<tr><td>Sim ground truth</td><td><b>%s</b></td><td>&nbsp;&nbsp;Decision acc (10 s)</td>"
            "<td><b>%.1f%%</b></td></tr>"
            "</table>" % (
                meta["claimed"] + 1, CLASS_COLORS[pred], CLASS_NAMES[pred], 100 * res["p"][pred],
                "#3ddc84" if res["sim"] >= SIM_THRESHOLD else "#ff4d4d", res["sim"], SIM_THRESHOLD,
                r["rssi_db"], res["jam_score"], r["latency_ms"], r["batch"],
                self.fps, self.frames, self.alerts, truth, acc))

    def on_state_change(self, res, meta):
        st = res["state"]
        claimed = meta["claimed"]
        self.set_status(st, claimed, "model warm-up %d%%" % self.train_pct)
        p = res["p"]
        if st == "AUTH":
            self.log("GREEN", "Authenticated: Device %d  P=%.2f  cos-sim=%.2f" % (claimed + 1, p[claimed], res["sim"]), "#3ddc84")
        elif st == "SPOOF":
            self.alerts += 1
            why = []
            if res["pred"] != claimed:
                why.append("classifier says %s (P=%.2f)" % (CLASS_NAMES[res["pred"]], p[res["pred"]]))
            if p[claimed] < AUTH_PROB:
                why.append("P(Device %d)=%.2f &lt; %.2f" % (claimed + 1, p[claimed], AUTH_PROB))
            if res["sim"] < SIM_THRESHOLD:
                why.append("cos-sim=%.2f &lt; %.2f" % (res["sim"], SIM_THRESHOLD))
            self.log("RED", "IMPOSTOR on Device %d channel: %s" % (claimed + 1, "; ".join(why) or "fingerprint mismatch"),
                     "#ff4d4d")
        elif st == "JAM":
            self.alerts += 1
            self.log("AMBER", "JAMMING detected: P(jammer)=%.2f  RSSI-anomaly=%.2f  -> fingerprint auth suspended"
                     % (p[JAMMER], res["rssi_score"]), "#ffb000")

    def closeEvent(self, ev):
        for th in (self.gen, self.inf, self.trainer):
            if th is not None:
                th.stop()
        for th in (self.gen, self.inf, self.trainer):
            if th is not None:
                th.wait(3000)
        ev.accept()


# =============================================================================
# 7. ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="RF-SENTINEL deep RF fingerprinting IDS")
    parser.add_argument("--retrain", action="store_true", help="ignore cached weights and re-bootstrap")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    parser.add_argument("--train-steps", type=int, default=0, help="bootstrap training steps (0 = auto)")
    args, qt_args = parser.parse_known_args()

    device, device_name = select_device(args.cpu)
    is_cuda = device.type == "cuda"
    steps = args.train_steps or (2500 if is_cuda else 1200)
    batch = 256 if is_cuda else 128
    print("=" * 78)
    print(" RF-SENTINEL  |  torch %s  |  CUDA available: %s" % (torch.__version__, torch.cuda.is_available()))
    print(" Compute backend: %s" % device_name)
    print("=" * 78)

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="#0b0f14", foreground="#c9d1d9")
    app = QtWidgets.QApplication([sys.argv[0]] + qt_args)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    hub = ModelHub(device)
    loaded = False if args.retrain else load_checkpoint(hub)
    if loaded:
        print(" Loaded cached weights:", CKPT_PATH)
    else:
        print(" Bootstrap training: %d steps x batch %d (inference already live)" % (steps, batch))

    win = MainWindow(hub, device_name, steps, batch, loaded)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

