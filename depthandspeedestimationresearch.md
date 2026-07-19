# Metric Depth Estimation & Vehicle Speed Estimation — Deep-Dive Research Report

**For:** Pathfinder — Embedded Lane-Keeping / Steer-by-Wire Autonomous Bicycle (UC Merced)
**Compiled:** July 2026
**Scope:** Models that output *absolute metric depth* ("this pixel is 10 m away"), ML methods that estimate *vehicle speed* ("that car is doing 22 mph"), and *public datasets* so you do NOT have to build your own.

> ⚠️ **Note on the example paper.** The PDF in this repo (`2506.05909v1.pdf`) is
> *"Twenty-Five Years of the Intelligent Driver Model"* (arXiv:2506.05909) — a survey of
> the IDM **car-following model** (traffic physics: desired speed, gap, acceleration).
> It is **not** a depth-estimation paper and it does not introduce a perception dataset.
> It is still useful to Pathfinder as the *planning/control* layer (IDM is the classic way to
> convert "car ahead at distance d, closing speed Δv" into a safe acceleration command),
> but everything below addresses the *perception* problem you described: getting `d` and `Δv`
> from cameras with machine learning.

---

## Table of Contents
1. [TL;DR — What Pathfinder should actually use](#1-tldr)
2. [Background: why metric depth is hard, and the 5 ways to get true meters](#2-background)
3. [Metric monocular depth models (the "this pixel is 10 m" models)](#3-metric-monocular-depth-models)
4. [Self-supervised depth + scale recovery](#4-self-supervised-depth--scale-recovery)
5. [Stereo depth (you already have the right hardware)](#5-stereo-depth)
6. [LiDAR + camera depth completion](#6-depth-completion)
7. [Per-object distance estimation (lightweight, pairs with YOLOv11)](#7-per-object-distance-estimation)
8. [Vehicle speed estimation with ML](#8-vehicle-speed-estimation)
9. [Real-time / Jetson-class deployment](#9-edge-deployment)
10. [DATASETS — the master catalog](#10-datasets)
11. [Recommended architectures for Pathfinder (3 tiers)](#11-recommended-architectures)
12. [Master bibliography index](#12-master-bibliography)

---

<a name="1-tldr"></a>
## 1. TL;DR — What Pathfinder should actually use

**You do not need to create your own dataset**, and you do not need a giant model. Concretely:

1. **Dense metric depth, one camera:** Use a *metric* head of a depth foundation model —
   **Depth Anything V2 Metric-Outdoor-Small** (HF: `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`),
   **Metric3D v2**, **UniDepth v2**, or **Depth Anything 3 Metric** — exported to TensorRT FP16.
   DA3-Small runs **23–43 FPS at 518×518 on a Jetson Orin NX with TensorRT**; expect similar or slightly
   lower on the Orin Nano Super. These models output depth **in meters per pixel, zero-shot**.
2. **Trust-but-verify scale with your stereo camera:** stereo gives geometric metric depth
   (Z = f·B/disparity) with no learning required; use it to sanity-check / rescale the monocular
   model in-domain. An OAK-D-class stereo unit is accurate to roughly 2 % out to ~4–8 m and usable to ~20–35 m.
3. **Speed of other vehicles:** don't regress speed from pixels directly. Use the standard,
   well-validated pipeline: **YOLOv11 detect → ByteTrack/OC-SORT track → per-box metric depth →
   Kalman filter on 3D position → velocity = filtered dZ/dt (+ ego-speed from wheel encoder)**.
   This is exactly what the strongest traffic-camera and onboard systems do, and errors of
   **±1–3 km/h** are reported in the literature with calibrated cameras.
4. **Fine-tuning data (only if zero-shot isn't accurate enough on your streets):**
   **KITTI (Eigen split) + Virtual KITTI 2 + DDAD + DrivingStereo** — all free, all with metric
   ground truth. For speed ground truth: **BrnoCompSpeed, AI City 2018 Track 1, TuSimple velocity,
   comma2k19**. Section 10 lists ~45 datasets.
5. **Ego-speed of the bike:** a $5 wheel/motor hall sensor beats every vision method; keep the
   vision-based ego-speed papers (Sec. 8.1) as a fallback/cross-check only.

Accuracy expectations to put in your paper: zero-shot monocular metric depth ≈ **5–15 % absolute
relative error** outdoors (AbsRel 0.05–0.10 on KITTI for SOTA fine-tuned; ~2× worse zero-shot on
unseen cameras); stereo error grows **quadratically** with distance; differentiating depth for
velocity amplifies noise, hence the Kalman filter is mandatory.

---

<a name="2-background"></a>
## 2. Background: why "exact meters" is the hard part

A single image is scale-ambiguous: a toy car at 1 m and a real car at 30 m can produce identical
pixels. Most famous depth models (MiDaS, DPT, Marigold, Depth Anything relative heads) therefore
output **relative/affine-invariant depth** — great-looking maps, but *not meters*. There are exactly
five ways the literature gets true metric scale:

| Route | Idea | Representative work |
|---|---|---|
| **A. Supervised metric training** | Train on LiDAR/stereo GT in meters for one domain/camera | DORN, BTS, AdaBins, NeWCRFs |
| **B. Camera-aware foundation models** | Condition on (or estimate) intrinsics so one model transfers metric scale across cameras | Metric3D, UniDepth, Depth Pro, ZeroDepth, DMD |
| **C. Multi-view geometry** | Stereo baseline or SfM with known translation gives scale for free | FoundationStereo, RAFT-Stereo, OAK-D/RealSense |
| **D. Sensor fusion** | Sparse LiDAR/radar points anchor the scale, network densifies | Depth completion (PENet, CompletionFormer), CenterFusion |
| **E. Geometric priors** | Known camera height above ground / ground-plane / object size fixes scale | DNet, MonoPP, GVDepth, Gamma-from-Mono, DisNet |

Pathfinder has assets for C (stereo cam), D (LiDAR), and E (fixed camera height on the bike) — you
are in an unusually good position to *verify* any model-B network online.

Key metrics used everywhere below: **AbsRel** (|d−d*|/d*), **RMSE [m]**, **δ<1.25** (% pixels within
25 % of GT), **SILog** (KITTI leaderboard metric).

---

<a name="3-metric-monocular-depth-models"></a>
## 3. Metric monocular depth models — "this pixel is X meters"

### 3.1 The zero-shot metric foundation models (2023–2026) — the direct answer to your question

| Model | Year/Venue | How it gets meters | Notes for Pathfinder |
|---|---|---|---|
| **ZoeDepth** [[arXiv:2302.12288](https://arxiv.org/abs/2302.12288)] | 2023 | MiDaS relative backbone + metric bins heads fine-tuned on NYU+KITTI | First good relative→metric bridge; needs fine-tune if your camera differs a lot |
| **ZeroDepth / PackNet-Zero** [[arXiv:2306.17253](https://arxiv.org/abs/2306.17253)] | ICCV 2023 | Input-level camera embeddings; scale-aware zero-shot | From Toyota Research; strong on driving data |
| **Metric3D** [[arXiv:2307.10984](https://arxiv.org/abs/2307.10984)] | ICCV 2023 | Canonical camera transform (rescale to canonical focal length) | Solves metric ambiguity across 1000s of cameras |
| **Metric3D v2** [[arXiv:2404.15506](https://arxiv.org/abs/2404.15506)] | TPAMI 2024 | Same + normals, 16 M images, thousands of cameras | Ranked #1 on several zero-shot metric benchmarks; heavy |
| **UniDepth** [[arXiv:2403.18913](https://arxiv.org/abs/2403.18913)] | CVPR 2024 | Predicts camera + depth jointly via pseudo-spherical output | No intrinsics needed at test time |
| **UniDepth v2** [[arXiv:2502.20110](https://arxiv.org/abs/2502.20110)] | 2025 | Simpler arch, edge-guided loss, uncertainty output | Uncertainty map is exactly what a safety filter on the ESP32 hub wants |
| **Depth Anything (V1)** [[arXiv:2401.10891](https://arxiv.org/abs/2401.10891)] | CVPR 2024 | 62 M unlabeled images; metric via NYU/KITTI fine-tune | |
| **Depth Anything V2 + Metric-Outdoor heads** [[arXiv:2406.09414](https://arxiv.org/abs/2406.09414)] | NeurIPS 2024 | Synthetic-GT teacher→student; metric heads fine-tuned on Virtual KITTI 2 (L1 loss in meters, 0–80 m) | **Best accuracy/latency trade-off for you**; S/B/L variants on HuggingFace |
| **Depth Pro (Apple)** [[arXiv:2410.02073](https://arxiv.org/abs/2410.02073)] | ICLR 2025 | Multi-scale ViT + **estimates focal length from the image itself** | 2.25 MP metric depth in 0.3 s on desktop GPU — too heavy for Orin Nano, great offline GT generator |
| **DMD (Google)** [[arXiv:2312.13252](https://arxiv.org/abs/2312.13252)] | 2023 | FOV-conditioned diffusion for zero-shot metric depth | Diffusion = slow; research reference |
| **Marigold** [[arXiv:2312.02145](https://arxiv.org/abs/2312.02145)] | CVPR 2024 | Diffusion prior (affine-invariant, *not* metric) | Include for completeness; needs route-E scaling |
| **BetterDepth** [[arXiv:2407.17952](https://arxiv.org/abs/2407.17952)] | 2024 | Plug-in diffusion refiner over metric models | |
| **PatchFusion** [[arXiv:2312.02284](https://arxiv.org/abs/2312.02284)] | CVPR 2024 | Tile-based high-res framework over ZoeDepth | |
| **MoGe / MoGe-2 (Microsoft)** [[arXiv:2410.19115](https://arxiv.org/abs/2410.19115), [GitHub](https://github.com/microsoft/MoGe)] | 2024/2025 | Point-map prediction; MoGe-2 adds **metric scale** + sharp detail | Strong open-source alternative |
| **Depth Anything 3 (ByteDance Seed)** [[GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3), [project](https://depth-anything-3.github.io/)] | Nov 2025 | Plain DINO transformer, depth-ray representation; **DA3-Metric series** (`metric_depth = focal · net_output / 300`) | DA3-Small: 23–43 FPS on Orin NX TensorRT FP16 — the current best "runs on Jetson, outputs meters" option |
| **MetricAnything** [[arXiv:2601.22054](https://arxiv.org/abs/2601.22054)] | 2026 | Scales metric pretraining with noisy heterogeneous sources | Newest entry; watch this space |
| **UniDAC** [[arXiv:2603.27105](https://arxiv.org/abs/2603.27105)] | 2026 | Universal metric depth for *any camera* incl. fisheye | Relevant if you add wide-FOV cams |
| **GVDepth** [[arXiv:2412.06080](https://arxiv.org/abs/2412.06080)] | 2024 | Zero-shot metric for **ground vehicles** via probabilistic fusion of object-size & ground-plane cues, single dataset training | Philosophically closest to a bicycle platform |
| **Survey: Monocular *Metric* Depth Estimation** [[arXiv:2501.11841](https://arxiv.org/abs/2501.11841)] | 2025 | — | Read this first; also MDPI version (Computers 14(11):502) |
| **MapAnything eval of MMDE for urban localization** [[arXiv:2509.14839](https://arxiv.org/abs/2509.14839)] | 2025 | — | Independent accuracy audit of the models above |

### 3.2 Classic supervised metric models (single-domain, still useful as fine-tuned baselines)

These are trained per-dataset (KITTI in meters) — accurate in-domain, brittle across cameras:

- **Eigen & Fergus**, first deep metric depth net — [arXiv:1406.2283](https://arxiv.org/abs/1406.2283) (NeurIPS 2014), multi-scale CNN; and [arXiv:1411.4734](https://arxiv.org/abs/1411.4734).
- **FCRN** (Laina et al., 2016) — [arXiv:1606.00373](https://arxiv.org/abs/1606.00373).
- **Kuznietsov et al.**, semi-supervised metric depth — [arXiv:1702.02706](https://arxiv.org/abs/1702.02706).
- **DORN** — depth as ordinal regression, CVPR 2018 — [arXiv:1806.02446](https://arxiv.org/abs/1806.02446).
- **BTS** — local planar guidance — [arXiv:1907.10326](https://arxiv.org/abs/1907.10326).
- **AdaBins** — adaptive depth bins, CVPR 2021 — [arXiv:2011.14141](https://arxiv.org/abs/2011.14141).
- **BinsFormer** — [arXiv:2204.00987](https://arxiv.org/abs/2204.00987).
- **NeWCRFs** — neural window FC-CRFs, CVPR 2022 — [arXiv:2203.01502](https://arxiv.org/abs/2203.01502).
- **PixelFormer** (WACV 2023) — [arXiv:2210.09071](https://arxiv.org/abs/2210.09071).
- **VA-DepthNet** (ICLR 2023) — [arXiv:2302.06556](https://arxiv.org/abs/2302.06556).
- **iDisc** — internal scene discretization, CVPR 2023 — [arXiv:2304.06334](https://arxiv.org/abs/2304.06334).
- **DPT** — dense prediction transformers (relative, but ancestor of everything above) — [arXiv:2103.13413](https://arxiv.org/abs/2103.13413); **MiDaS** — [arXiv:1907.01341](https://arxiv.org/abs/1907.01341).
- **SideRT** — real-time transformer depth — [arXiv:2204.13892](https://arxiv.org/abs/2204.13892).
- **NVS-MonoDepth** — [arXiv:2112.12577](https://arxiv.org/abs/2112.12577).
- **PackNet-SAN** — unified prediction + completion — [arXiv:2103.16690](https://arxiv.org/abs/2103.16690).

State of the art in-domain on KITTI (Eigen): AbsRel ≈ 0.05, RMSE ≈ 2.0–2.2 m over 0–80 m —
i.e., a car at 30 m is typically located within ±1.5–3 m; a fine-tuned model on your own camera
does considerably better at short range (your safety-critical envelope at ≤24 mph is ~0–30 m).

### 3.3 Video / temporally-consistent depth (helps speed estimation directly)

- **Video Depth Anything** (CVPR 2025 Highlight) — [arXiv:2501.12375](https://arxiv.org/abs/2501.12375), [project](https://videodepthanything.github.io/) — consistent depth for arbitrarily long video.
- **Online Video Depth Anything** — streaming, low-memory — [arXiv:2510.09182](https://arxiv.org/abs/2510.09182).
- **DA3-Streaming** — sliding-window ultra-long video inference under 12 GB ([DA3 repo](https://github.com/ByteDance-Seed/Depth-Anything-3)).
- **DepthCrafter** — video diffusion depth — [arXiv:2409.02095](https://arxiv.org/abs/2409.02095).
- **ChronoDepth** — [arXiv:2406.01493](https://arxiv.org/abs/2406.01493).
- **AsyncMDE** — asynchronous spatial memory, 161 FPS on AGX Orin — [arXiv:2603.10438](https://arxiv.org/abs/2603.10438).

Temporal consistency matters for you because **speed = d(depth)/dt**: frame-to-frame depth jitter
becomes velocity noise. A temporally-consistent model + Kalman filtering is the fix.

---

<a name="4-self-supervised-depth--scale-recovery"></a>
## 4. Self-supervised depth + scale recovery (train on *your* bike videos with no labels)

Self-supervised methods learn depth from raw monocular video via view synthesis — attractive
because you could train on Pathfinder's own recordings. Caveat: they are scale-ambiguous *unless*
you add one of the scale signals below (you have all of them).

**Core line of work:**
- **SfMLearner** (Zhou et al., CVPR 2017) — [arXiv:1704.07813](https://arxiv.org/abs/1704.07813)
- **Monodepth** (Godard, CVPR 2017) — [arXiv:1609.03677](https://arxiv.org/abs/1609.03677)
- **Monodepth2** (ICCV 2019) — [arXiv:1806.01260](https://arxiv.org/abs/1806.01260) — still the standard baseline
- **SC-Depth** — scale-consistent ego-motion — [arXiv:1908.10553](https://arxiv.org/abs/1908.10553)
- **PackNet-SfM** (CVPR 2020) — [arXiv:1905.02693](https://arxiv.org/abs/1905.02693) — **key paper: adds weak *velocity* supervision (from CAN/GPS speed) → metric-scaled depth**. Your e-bike's wheel speed sensor is exactly this signal.
- **ManyDepth** — multi-frame cost volumes — [arXiv:2104.14540](https://arxiv.org/abs/2104.14540)
- **DepthFormer / multi-frame transformer** (TRI) — [arXiv:2204.07616](https://arxiv.org/abs/2204.07616)
- **DynamicDepth** — handles moving objects — [arXiv:2203.15174](https://arxiv.org/abs/2203.15174)
- **MonoViT** — [arXiv:2208.03543](https://arxiv.org/abs/2208.03543); **Lite-Mono** — [arXiv:2211.13202](https://arxiv.org/abs/2211.13202) (both light enough for Jetson)
- **MonoDEVSNet** — virtual-world supervision + real SfM — [arXiv:2103.12209](https://arxiv.org/abs/2103.12209)

**Scale-recovery specific (route E):**
- **DNet** — camera-height-based scale recovery for driving — [arXiv:2004.05560](https://arxiv.org/abs/2004.05560)
- **"Camera Height Doesn't Change"** — road-scene scale-aware depth — [arXiv:2312.04530](https://arxiv.org/abs/2312.04530)
- **MonoPP** — metric scale from planar-parallax + known camera height (automotive) — [arXiv:2411.19717](https://arxiv.org/abs/2411.19717)
- **Gamma-from-Mono** — road-relative metric self-supervised geometry — [arXiv:2512.04303](https://arxiv.org/abs/2512.04303)
- **VI rescaling for aerial nav** (visual-inertial metric rescaling of zero-shot depth) — [arXiv:2509.08159](https://arxiv.org/abs/2509.08159) — same trick works with your IMU.

**Practical takeaway:** mount height of your front camera is fixed and known (± suspension). A
ground-plane homography from that height alone converts *any* relative depth map into meters on the
road surface — a 20-line, zero-FLOP scale calibration you should implement regardless of model choice.

---

<a name="5-stereo-depth"></a>
## 5. Stereo depth — your depth camera already speaks meters

Depth from stereo is metric by construction: `Z = f·B / disparity` (f = focal px, B = baseline m).
Error grows quadratically: ΔZ ≈ Z²·Δd/(f·B) — fine at 5–30 m with your baseline, poor at 100 m.

**Learned stereo matching (if you process raw stereo pairs on the Jetson):**
- **PSMNet** — [arXiv:1803.08669](https://arxiv.org/abs/1803.08669); **GwcNet** — [arXiv:1903.04025](https://arxiv.org/abs/1903.04025); **GA-Net** — [arXiv:1904.06587](https://arxiv.org/abs/1904.06587); **AANet** — [arXiv:2004.09548](https://arxiv.org/abs/2004.09548)
- **HITNet** (Google, real-time) — [arXiv:2007.12140](https://arxiv.org/abs/2007.12140)
- **RAFT-Stereo** — [arXiv:2109.07547](https://arxiv.org/abs/2109.07547); **CREStereo** — [arXiv:2203.11483](https://arxiv.org/abs/2203.11483); **ACVNet** — [arXiv:2203.02146](https://arxiv.org/abs/2203.02146); **IGEV-Stereo** — [arXiv:2303.06615](https://arxiv.org/abs/2303.06615)
- **MobileStereoNet** — embedded-friendly — [arXiv:2108.09770](https://arxiv.org/abs/2108.09770)
- **StereoAnything** — large-scale mixed-data stereo foundation — [arXiv:2411.14053](https://arxiv.org/abs/2411.14053)
- **FoundationStereo** (NVIDIA, CVPR 2025 Best-Paper nominee) — zero-shot stereo foundation model, 1 M synthetic pairs — [arXiv:2501.09898](https://arxiv.org/abs/2501.09898), [GitHub](https://github.com/NVlabs/FoundationStereo)
- **Fast-FoundationStereo** — real-time distilled variant — [arXiv:2512.11130](https://arxiv.org/abs/2512.11130)

**Hardware reality check** (from the 2025 empirical comparison [arXiv:2501.07421](https://arxiv.org/abs/2501.07421) and vendor docs): RealSense D435/D455 ≈ <2 % error at 2–4 m; OAK-D Pro perceives to ~35 m with on-device NN compute (offloads your Jetson). For a ≤24 mph bike whose braking envelope is ≲25 m, **a good stereo camera alone already meets the "exact meters" requirement in the near field** — the monocular network's job is the 25–80 m band and robustness (rain on one lens, etc.).

---

<a name="6-depth-completion"></a>
## 6. Depth completion — you have a LiDAR; use it to anchor scale

Task: sparse LiDAR points + RGB → dense metric depth. This is route D and the most accurate option
of all (KITTI DC leaderboard RMSE ≈ 0.7 m over 0–80 m).

- **Sparsity-Invariant CNNs** (defined the KITTI Depth Completion benchmark, 3DV 2017) — [arXiv:1708.06500](https://arxiv.org/abs/1708.06500)
- **Fast depth completion on CPU** — [arXiv:1802.00036](https://arxiv.org/abs/1802.00036) (runs on the ESP32-adjacent budget!)
- **Depth-Normal constraints** (ICCV 2019) — [arXiv:1910.06727](https://arxiv.org/abs/1910.06727)
- **NLSPN** — non-local spatial propagation — [arXiv:2007.10042](https://arxiv.org/abs/2007.10042)
- **PENet** — [arXiv:2103.00783](https://arxiv.org/abs/2103.00783); **EfficientPENet** (real-time) — [arXiv:2604.18790](https://arxiv.org/abs/2604.18790)
- **CompletionFormer** — [arXiv:2304.13030](https://arxiv.org/abs/2304.13030)
- **SemAttNet** — semantic-aware — [arXiv:2204.13635](https://arxiv.org/abs/2204.13635)
- **Planar-LiDAR + mono fusion** (2D scanner, very close to your hardware class) — [arXiv:2009.01875](https://arxiv.org/abs/2009.01875)
- **CU-Net** LiDAR-only completion — [arXiv:2210.14898](https://arxiv.org/abs/2210.14898)
- **Comprehensive survey of depth completion** — [MDPI Sensors 22(18):6969](https://www.mdpi.com/1424-8220/22/18/6969)
- **Prompt-style fusion:** *Prompt Depth Anything* — LiDAR prompting of a depth foundation model — [arXiv:2412.14015](https://arxiv.org/abs/2412.14015)

Your LiDAR currently covers left/right/rear only; even a few hundred points per frame in the side
views can continuously *calibrate the monocular model's scale* for the front view.

---

<a name="7-per-object-distance-estimation"></a>
## 7. Per-object distance estimation — cheapest path to "that car is 10 m away"

You already run YOLOv11. Instead of dense depth, regress one distance number per detected object:

- **DisNet** (IROS PPNIV 2018) — MLP on bounding-box geometry (h, w, diagonal, class prior size) → distance; trained with laser GT — [paper PDF](https://project.inria.fr/ppniv18/files/2018/10/paper22.pdf). Trivially runs on the ESP32, ~1 % of your compute budget.
- **Learning Object-Specific Distance** (Zhu & Fang, ICCV 2019) — end-to-end per-object distance with keypoint projection; introduced object-distance annotations on KITTI/nuScenes — [arXiv:1909.04182](https://arxiv.org/abs/1909.04182)
- **Dist-YOLO** — YOLO extended with a distance term in the head/loss (shares backbone; near-zero extra latency) — [MDPI Appl. Sci. 12(3):1354](https://www.mdpi.com/2076-3417/12/3/1354). **This is the single most direct "solution" for your current stack: add a distance channel to YOLOv11's head.**
- **Obstacle detection + bbox-distance for railways** — [ResearchGate](https://www.researchgate.net/publication/374976238) (DisNet lineage)
- **Simultaneous detection + distance for indoor AVs** — [MDPI Electronics 12(23):4719](https://www.mdpi.com/2079-9292/12/23/4719)
- **Vehicle distance from monocular camera for ADAS** — [2022, MDPI/ResearchGate](https://www.researchgate.net/publication/366334407)
- **Geometry route (no ML):** flat-ground assumption + intrinsics: `Z = f·H_cam / (y_bottom − y_horizon)` — the classic Mobileye recipe; combine with class-prior width for redundancy. Works because your camera height is fixed.
- **Pseudo-LiDAR** — lift monocular/stereo depth to point clouds for 3D detection — [arXiv:1812.07179](https://arxiv.org/abs/1812.07179)
- **Monocular 3D detection** (gives full 3D box incl. center distance): **SMOKE** [arXiv:2002.10111](https://arxiv.org/abs/2002.10111), **FCOS3D** [arXiv:2104.10956](https://arxiv.org/abs/2104.10956), **DETR3D** [arXiv:2110.06922](https://arxiv.org/abs/2110.06922), **BEVDet** [arXiv:2112.11790](https://arxiv.org/abs/2112.11790), **BEVFormer** [arXiv:2203.17270](https://arxiv.org/abs/2203.17270). nuScenes-trained variants of these also emit **per-object velocity** (see 8.2).

---

<a name="8-vehicle-speed-estimation"></a>
## 8. Vehicle speed estimation with ML — "the car is doing 22 mph"

Three distinct problems; pick the row that matches the sentence you wrote:

| Problem | Camera | Output | Sections |
|---|---|---|---|
| Ego-speed | on the bike | bike's own mph | 8.1 |
| Other vehicles, from the bike | moving, onboard | relative + absolute mph of cars | 8.2 |
| Other vehicles, roadside | static, surveillance | absolute mph of cars | 8.3 |

### 8.1 Ego-speed from onboard video
- **comma.ai speed challenge** — predict CAN-bus speed from dashcam; community solutions use dense optical flow + CNN (NVIDIA PilotNet-style) — [example repo](https://github.com/jancervenka/speed-challenge)
- **3DCMA** — 3D convolution with masked attention for ego-speed — [SSAD 2023 paper](https://trust-ai.github.io/SSAD2023/assets/papers/7.pdf)
- **Ego-speed via optical flow analysis** (2025) — [ResearchGate](https://www.researchgate.net/publication/390086658)
- **RNN/Transformer video speed estimation** — [arXiv:2502.15545](https://arxiv.org/abs/2502.15545)
- **Speed estimation on KITTI from motion + monocular depth** — [arXiv:1907.06989](https://arxiv.org/abs/1907.06989)
- Optical-flow backbones used by all of the above: **FlowNet** [arXiv:1504.06852](https://arxiv.org/abs/1504.06852), **FlowNet2** [arXiv:1612.01925](https://arxiv.org/abs/1612.01925), **PWC-Net** [arXiv:1709.02371](https://arxiv.org/abs/1709.02371), **RAFT** [arXiv:2003.12039](https://arxiv.org/abs/2003.12039)
- Verdict for Pathfinder: use the wheel-speed sensor; vision ego-speed is a research toy here.

### 8.2 Speed of *other* vehicles from a moving platform (your actual use case)
- **Kampelmühler et al.** — winner, CVPR 2017 TuSimple velocity challenge; MLP on tracked-box trajectory features beats deep flow/depth features at long range — [arXiv:1802.07094](https://arxiv.org/abs/1802.07094)
- **TuSimple Velocity Estimation benchmark** — 1 000+ 2-s clips, radar-GT velocity/position, Near/Medium/Far protocol — [GitHub](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/velocity_estimation)
- **Song et al., ICRA 2020** — end-to-end inter-vehicle **distance + relative velocity** from two consecutive monocular frames; vehicle-centric sampling to undo perspective distortion; lightweight — [arXiv:2006.04082](https://arxiv.org/abs/2006.04082), [code](https://github.com/ZhenboSong/mono_velocity). **Closest single paper to "after prediction it says 10 m and 22 mph" from a moving camera.**
- **Monocular Quasi-Dense 3D Object Tracking** — full 3D tracks (⇒ velocities) on KITTI/nuScenes/Waymo — [arXiv:2103.07351](https://arxiv.org/abs/2103.07351)
- **CenterTrack** — detection + 2D/3D displacement (velocity head on nuScenes) — [arXiv:2004.01177](https://arxiv.org/abs/2004.01177)
- **CenterFusion** — radar+camera fusion; big velocity-accuracy win *without temporal info* — [arXiv:2011.04841](https://arxiv.org/abs/2011.04841); **CFTrack** — [arXiv:2107.05150](https://arxiv.org/abs/2107.05150)
- **Full-Velocity Radar Returns by Radar-Camera Fusion** (ICCV 2021) — [arXiv:2108.10453](https://arxiv.org/abs/2108.10453)
- **DeepCrashTest** — dashcam video → virtual crash tests (extracts trajectories + speeds) — [arXiv:2003.11766](https://arxiv.org/abs/2003.11766) — directly relevant to your "simulating car crash scenarios" future-work bullet.
- **nuScenes-style 3D detectors** (FCOS3D, BEVFormer, BEVDet — Sec. 7) are trained to output a **velocity vector in m/s per object** because the nuScenes benchmark scores AVE (average velocity error). Fine-tuning any of them = a literal "car speed" regressor.

### 8.3 Roadside/surveillance speed measurement (mature; where the ±1 km/h numbers come from)
- **Survey (start here):** *Vision-based vehicle speed estimation: a survey* — Fernández Llorca et al., IET ITS 2021 — reviews **135+ papers** — [arXiv:2101.06159](https://arxiv.org/abs/2101.06159)
- **Dubská et al., BMVC 2014** — fully-automatic camera calibration from vanishing points for speed measurement (foundational)
- **Sochor et al.** — **BrnoCompSpeed**: 18×1 h FHD videos, 20 865 vehicles, LiDAR-gate GT speeds; the reference benchmark — [arXiv:1702.06441](https://arxiv.org/abs/1702.06441), [dataset page](https://medusa.fit.vutbr.cz/traffic/research-topics/traffic-camera-calibration/brnocompspeed/)
- **Luvizon et al.** — license-plate detection + tracking; inductive-loop GT; −0.5 km/h mean error, 96 % within [−3,+2] km/h — [SIBGRAPI 2016 PDF](http://sibgrapi.sid.inpe.br/col/sid.inpe.br/sibgrapi/2016/08.16.09.03/doc/sibgrapi_2016_luvizon.pdf)
- **Revaud & Humenberger (NAVER), ICCV 2021** — robust automatic monocular speed estimation, no manual calibration — [CVF open access](https://openaccess.thecvf.com/content/ICCV2021/papers/Revaud_Robust_Automatic_Monocular_Vehicle_Speed_Estimation_for_Traffic_Surveillance_ICCV_2021_paper.pdf)
- **ISPRS 2020** — accurate speed from monocular footage — [paper](https://isprs-annals.copernicus.org/articles/V-2-2020/419/2020/)
- **NVIDIA AI City Challenge 2018 Track 1** — speed from 27 FHD videos, control-vehicle GT; UW team won with detection+tracking+calibration — [challenge](https://www.aicitychallenge.org/2018-data-sets/), [winning code](https://github.com/zhengthomastang/2018AICity_TeamUW), [overview paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Naphade_The_2018_NVIDIA_CVPR_2018_paper.pdf)
- **Traffic camera calibration via vehicle vanishing points** — [arXiv:2103.11438](https://arxiv.org/abs/2103.11438)
- **Efficient vision-based speed estimation** (2025) — [arXiv:2505.01203](https://arxiv.org/abs/2505.01203)
- **Modern YOLO pipelines:** YOLOv8 speed system [arXiv:2406.07710](https://arxiv.org/abs/2406.07710); YOLOv5s+DeepSORT with multi-sensor verification — [Frontiers in Physics 2024](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1371320/full); deep-homography rectification + YOLOv8 + ByteTrack (0.53 km/h abs error) — [ResearchGate 2024](https://www.researchgate.net/publication/386195390); line-homography multi-vehicle framework — [ETASR 2025](https://etasr.com/index.php/ETASR/article/view/19490)
- **Monocular speed, Scientific Reports 2025** — detection→tracking→3D positioning→speed; 97.6 % avg accuracy — [Nature s41598-025-87077-6](https://www.nature.com/articles/s41598-025-87077-6)
- **UAV-based speed** — [MDPI Algorithms 17(12):558](https://www.mdpi.com/1999-4893/17/12/558)
- **View-invariant speed from simulator data** — [arXiv:2206.00343](https://arxiv.org/abs/2206.00343)
- **Audio-video speed estimation + VS13 dataset** — [arXiv:2212.01651](https://arxiv.org/abs/2212.01651)
- **Occlusion-robust forensic speed** — [Springer MTAP 2026](https://link.springer.com/article/10.1007/s11042-026-21222-9)

### 8.4 Time-to-collision (often the better safety quantity for a bike)
TTC = Z/(dZ/dt) needs *no absolute scale at all* (ratios cancel) — robust even with relative depth:
- **Forecasting TTC from monocular video** — [arXiv:1903.09102](https://arxiv.org/abs/1903.09102)
- **Binary TTC** (CVPR 2021, NVIDIA) — [arXiv:2101.04777](https://arxiv.org/abs/2101.04777)
- **Constant-jerk instantaneous TTC with monocular camera** — [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S2215098625000667)
- **Vision-based collision-warning systematic review** — [PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11856197/)
- **Cyclist orientation detection for VRU safety** — [arXiv:2004.11909](https://arxiv.org/abs/2004.11909)

### 8.5 The math you'll put in the paper
```
Z_t  = median metric depth inside tracked box (or Dist-YOLO output)   [m]
v_rel = Kalman-filtered dZ/dt  (state: [Z, Ż], constant-velocity model)
v_car = v_ego(wheel sensor) + v_rel(along optical axis, signed)        [m/s]
mph  = 2.23694 · v_car
TTC  = Z / max(−Ż, ε)
```
At 30 FPS with per-frame depth noise σ_Z ≈ 0.3 m, raw finite differences give σ_v ≈ 9 m/s — useless;
a 10-frame Kalman window brings it to ≈ 0.5–1 m/s (±1–2 mph), matching what BrnoCompSpeed-class
systems report. This is why every serious paper is detection→tracking→filtering, not per-frame regression.

---

<a name="9-edge-deployment"></a>
## 9. Real-time / Jetson-class deployment

- **FastDepth** (MIT, ICRA 2019) — 178 FPS on Jetson TX2 GPU — [arXiv:1903.03273](https://arxiv.org/abs/1903.03273)
- **RT-MonoDepth** — 18–30 FPS on original Jetson Nano, 253–364 FPS on AGX Orin — [arXiv:2308.10569](https://arxiv.org/abs/2308.10569)
- **Guided Decoding** — lightweight encoder-decoder for embedded — [arXiv:2203.04206](https://arxiv.org/abs/2203.04206)
- **Real-time human depth on embedded** — [arXiv:2108.10506](https://arxiv.org/abs/2108.10506)
- **ZipDepth** — 6.1 M params, 34 FPS @ 15 W Orin NX, zero-shot — [project](https://zipdepth.github.io/)
- **AsyncMDE** — 161 FPS TensorRT AGX Orin — [arXiv:2603.10438](https://arxiv.org/abs/2603.10438)
- **DA3 ROS 2 wrapper** — DA3-Small @ 518², TensorRT FP16: 23 FPS camera-limited / 43 FPS compute on Orin NX — [GitHub](https://github.com/GerdsenAI/GerdsenAI-Depth-Anything-3-ROS2-Wrapper) — **drop-in for your ROS 2 stack**
- **Fast-FoundationStereo** — real-time zero-shot stereo — [arXiv:2512.11130](https://arxiv.org/abs/2512.11130)
- **HITNet / MobileStereoNet** (Sec. 5) for stereo on-Jetson; or run stereo **on the OAK-D's Myriad X** and spend zero Jetson FLOPs.

Deployment recipe that matches your LaneATT+YOLOv11 pipeline: export DA-V2-Metric-Outdoor-**Small**
(24.8 M params) or DA3-Small-Metric to ONNX → TensorRT FP16 (INT8 after calibration), 384×640 input
to match your LaneATT resolution, run at 10–15 Hz on a dedicated CUDA stream (depth doesn't need
30 Hz; the Kalman filter interpolates), keep YOLOv11n at full rate.

---

<a name="10-datasets"></a>
## 10. DATASETS — the master catalog (you do not need to build one)

### 10.1 Driving datasets with **metric depth ground truth** (LiDAR/stereo)

| Dataset | Size / GT | Why use it | Link |
|---|---|---|---|
| **KITTI** (raw + Eigen split) | 61 scenes, Velodyne HD64 GT; the canonical metric-depth benchmark (0–80 m) | Every model above reports on it | [cvlibs.net](http://www.cvlibs.net/datasets/kitti/) · Geiger CVPR 2012 |
| **KITTI Depth Prediction / Completion** | 93k frames semi-dense GT (LiDAR+SGM accumulated); official leaderboards | Fine-tuning + fair evaluation | [arXiv:1708.06500](https://arxiv.org/abs/1708.06500) |
| **KITTI-360** | 320k images, fused dense GT, 360° | Successor with richer annotation | [arXiv:2109.13410](https://arxiv.org/abs/2109.13410) |
| **DDAD** (Toyota TRI) | 12 cams 360°, dense long-range LiDAR GT **to 250 m**, US+Japan | Long-range metric depth; official depth challenge | [GitHub TRI-ML/DDAD](https://github.com/TRI-ML/DDAD) |
| **DrivingStereo** | **182 188** stereo pairs with LiDAR-filtered disparity/depth PNGs | 100× KITTI stereo size; weather subsets | [site](https://drivingstereo-dataset.github.io/) · CVPR 2019 |
| **nuScenes** | 1 000 scenes, 32-beam LiDAR, **per-object velocity annotations** | Train depth *and* velocity heads | [arXiv:1903.11027](https://arxiv.org/abs/1903.11027) |
| **Waymo Open** | 1 150 scenes, 5 LiDARs, HD cams; velocity via tracks | Scale + diversity | [arXiv:1912.04838](https://arxiv.org/abs/1912.04838) |
| **Argoverse 2** | 1 000 lidar-annotated scenes, 4 M+ frames | Modern US streets | [arXiv:2301.00493](https://arxiv.org/abs/2301.00493) |
| **ApolloScape** | 140k frames, per-pixel depth for static scenes | Dense GT in Chinese cities | [arXiv:1803.06184](https://arxiv.org/abs/1803.06184) |
| **A2D2** (Audi) | 41k annotated frames, 5 LiDARs | Free commercial-friendly license | [arXiv:2004.06320](https://arxiv.org/abs/2004.06320) |
| **PandaSet** | 103 scenes, 64-beam + solid-state LiDAR | Free, modern sensors | [arXiv:2112.12610](https://arxiv.org/abs/2112.12610) |
| **ONCE** | 1 M scenes (144 h), LiDAR | Huge unlabeled pool for self-sup | [arXiv:2106.11037](https://arxiv.org/abs/2106.11037) |
| **Lyft Level 5** | 1 000+ h, LiDAR | Motion + perception | Houston et al., 2020 |
| **Cityscapes** | 5k fine frames + **SGM disparity maps** | Segmentation+depth combo | [arXiv:1604.01685](https://arxiv.org/abs/1604.01685) |
| **Mapillary Planet-Scale Depth (PSD)** | 750k+ street images, SfM metric depth, **thousands of cameras worldwide** | Camera diversity → zero-shot robustness | [ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470579.pdf) · [blog](https://blog.mapillary.com/update/2020/08/20/planet-scale-depth-dataset.html) |
| **ROVR Open Dataset** | 200k HD frames, highway/rural/urban, day/night/weather, 3 continents; crowd-collected LiDAR rigs | Newest large open depth set (target 1 M clips) | [arXiv:2508.13977](https://arxiv.org/abs/2508.13977) |
| **DENSE / Seeing Through Fog** | Fog/snow/rain, gated+LiDAR | Adverse weather depth | Bijelic et al., CVPR 2020 |
| **Oxford RobotCar / Radar RobotCar** | Year-long repeated route, LiDAR+radar | Seasonal robustness | Maddern et al., IJRR 2017 |
| **DIODE (outdoor split)** | Dense laser-scanner GT to 300 m | Very dense outdoor GT | [arXiv:1908.00463](https://arxiv.org/abs/1908.00463) |
| **Make3D** | 534 images + laser GT | Historic outdoor benchmark | Saxena et al., PAMI 2009 |
| **UASOL** | Large-baseline stereo urban walks | Pedestrian-height viewpoints (≈ bike height!) | Sci. Data 2019 |
| **SeasonDepth** | Cross-season depth benchmark | Robustness eval | [OpenReview](https://openreview.net/forum?id=kOxP7Fbeduy) |

### 10.2 Synthetic (perfect dense metric GT, free labels)

| Dataset | Notes | Link |
|---|---|---|
| **Virtual KITTI 2** | Photo-realistic KITTI clone, dense depth to 80 m; **what DA-V2's outdoor metric head is fine-tuned on** | [arXiv:2001.10773](https://arxiv.org/abs/2001.10773) |
| **SYNTHIA** | Urban synthetic + depth | Ros et al., CVPR 2016 |
| **CARLA** | Generate unlimited depth/speed GT for *bicycle-view* cameras — you can render your exact camera height/FOV | [arXiv:1711.03938](https://arxiv.org/abs/1711.03938) |
| **SHIFT** | 2.5 M synthetic driving frames, depth + weather/time shifts | [arXiv:2206.08367](https://arxiv.org/abs/2206.08367) |
| **SceneFlow** | 39k stereo pairs (stereo pretraining standard) | [arXiv:1512.02134](https://arxiv.org/abs/1512.02134) |
| **TartanAir** | Aerial/ground SLAM with depth | [arXiv:2003.14338](https://arxiv.org/abs/2003.14338) |
| **Hypersim** | Indoor synthetic (for generalist training mixes) | [arXiv:2011.02523](https://arxiv.org/abs/2011.02523) |
| **MPI Sintel** | Movie-derived depth/flow | Butler et al., ECCV 2012 |

### 10.3 Indoor / generalist metric GT (used in every foundation-model training mix)
**NYU Depth v2** (Silberman ECCV 2012, Kinect), **ScanNet** [arXiv:1702.04405](https://arxiv.org/abs/1702.04405), **SUN RGB-D** (CVPR 2015), **iBims-1**, **ETH3D**, **Middlebury 2014**, **DIML/CVL RGB-D**, **MegaDepth** [arXiv:1804.00607](https://arxiv.org/abs/1804.00607) (SfM, scale up to a factor), **ReDWeb / WSVD / OASIS** (relative-depth web data), **SYNS-Patches** (MDEC challenge benchmark — see [arXiv:2304.07051](https://arxiv.org/abs/2304.07051)).

### 10.4 Datasets with **speed ground truth** (for the "22 mph" half)

| Dataset | GT source | Contents | Link |
|---|---|---|---|
| **BrnoCompSpeed** | LiDAR optical gates + GPS | 18×1 h FHD, 20 865 vehicles, 6 sites | [arXiv:1702.06441](https://arxiv.org/abs/1702.06441) |
| **AI City Challenge 2018 Track 1** | Instrumented control vehicles | 27×1 min FHD @30fps | [aicitychallenge.org](https://www.aicitychallenge.org/2018-data-sets/) |
| **TuSimple Velocity** | Radar | 1 000+ 2-s onboard clips; Near/Med/Far eval | [GitHub](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/velocity_estimation) |
| **Luvizon / UTFPR** | Inductive loops | 5 h, 3-lane road, plates annotated | SIBGRAPI 2016 / IEEE TITS 2017 |
| **comma2k19** | CAN speed + GNSS/IMU | 33 h highway commute; **ego-speed** | [arXiv:1812.05752](https://arxiv.org/abs/1812.05752) · [HF dataset](https://huggingface.co/datasets/commaai/comma2k19) |
| **comma speed challenge** | CAN | dashcam→speed regression toy set | [GitHub](https://github.com/commaai/speedchallenge) |
| **VS13** | Cruise-control constant speeds | 400 audio-video clips, 13 vehicles, 30–105 km/h | [arXiv:2212.01651](https://arxiv.org/abs/2212.01651) |
| **nuScenes / Waymo / Argoverse 2** | Annotated 3D tracks | per-object velocity vectors (AVE metric) | see 10.1 |
| **KITTI raw OXTS** | RT3003 GNSS/INS | ego velocity per frame; object speeds via tracklets | cvlibs.net |
| **DGPS driving-simulator set (view-invariant speed)** | simulator | multi-view speed | [arXiv:2206.00343](https://arxiv.org/abs/2206.00343) |

### 10.5 What this means for Pathfinder
- **Zero-shot first:** DA-V2-Metric-Outdoor / DA3-Metric / Metric3Dv2 / UniDepthV2 need *no data from you at all*.
- **If fine-tuning:** KITTI + vKITTI2 + DrivingStereo (+ DDAD for long range) is the standard mix; ~1 weekend on a single A100/4090, then TensorRT-export.
- **Your only data-collection job** is a *validation* set, not a training set: ~200 frames from the bike with the OAK-D depth + a tape-measured cone course → report AbsRel/RMSE of the mono model on *your* camera in the paper. That's a table, not a dataset.
- Note: CULane (your LaneATT training set) has **no depth GT** — lane detection and depth stay separate supervision streams.

---

<a name="11-recommended-architectures"></a>
## 11. Recommended solution architectures for Pathfinder

### Tier A — "ship this month" (near-zero extra GPU cost)
```
OAK-D stereo depth (on-camera Myriad X) ──┐
YOLOv11n boxes ───────────────────────────┼→ median depth per box → ByteTrack ID
wheel-speed sensor (ego v) ───────────────┘→ Kalman [Z, Ż] → distance (m), speed (mph), TTC
```
- Metric by construction, ~0 additional Jetson FLOPs, works to ~25–35 m (covers the ≤24 mph envelope).
- Literature basis: Dist-YOLO / DisNet (per-box distance), Song ICRA 2020 (distance+velocity), BrnoCompSpeed-style tracking+filtering.

### Tier B — dense metric depth (the "research contribution" tier)
- DA-V2-Metric-Outdoor-Small or DA3-Small-Metric, TensorRT FP16, 384×640, 10–15 Hz.
- Online scale check: compare mono depth vs stereo depth on the 5–20 m band each frame; apply a slow EMA scale correction (route-E ground-plane check as backup).
- Feeds: obstacle map for the lane-keeping planner + per-object distances for Tier A logic at longer range (25–80 m).

### Tier C — fusion (future work section of your paper)
- Side/rear LiDAR → depth-completion-style scale anchoring (Prompt Depth Anything / EfficientPENet).
- Optional cheap automotive radar (e.g., CAN radar) → CenterFusion-style velocity without differentiation — radar measures Doppler velocity *directly*, which is why AV stacks love it.
- IDM (your example paper!) as the longitudinal controller consuming (d, Δv): a_IDM = f(v, Δv, d).

### Numbers to target/report (grounded in the literature)
| Quantity | Achievable | Source basis |
|---|---|---|
| Depth error @10 m (stereo) | ±0.1–0.3 m | RealSense/OAK spec, arXiv:2501.07421 |
| Depth error @30 m (mono metric, fine-tuned) | ±1.5–3 m (5–10 %) | KITTI SOTA AbsRel ≈ 0.05 |
| Depth error @30 m (mono metric, zero-shot) | ±3–6 m (10–20 %) | Metric3D/UniDepth zero-shot evals |
| Vehicle speed error (tracked, calibrated) | ±1–3 km/h | Luvizon, BrnoCompSpeed, homography+YOLOv8 (0.53 km/h) |
| Runtime (DA3-Small metric, Orin-class, TRT FP16) | 20–40 FPS @518² | DA3 ROS 2 wrapper benchmarks |

---

<a name="12-master-bibliography"></a>
## 12. Master bibliography index (by section)

**Count:** ~135 distinct works cited above. Quick index:

- **Surveys (7):** MMDE survey 2501.11841 · Llorca speed survey 2101.06159 (135 papers reviewed) · Depth-completion survey (Sensors 2022) · MDEC challenge 2304.07051 · Collision-warning review (PMC 2025) · Monocular depth eval critique 2510.19814 · MapAnything MMDE audit 2509.14839
- **Zero-shot metric mono (18):** ZoeDepth · ZeroDepth · Metric3D · Metric3Dv2 · UniDepth · UniDepthV2 · DA v1 · DA v2 (+metric heads) · DA3 (+metric/streaming) · Depth Pro · DMD · Marigold · BetterDepth · PatchFusion · MoGe/MoGe-2 · MetricAnything · UniDAC · GVDepth
- **Supervised metric mono (16):** Eigen'14 ×2 · FCRN · Kuznietsov · DORN · BTS · AdaBins · BinsFormer · NeWCRFs · PixelFormer · VA-DepthNet · iDisc · MiDaS · DPT · SideRT · NVS-MonoDepth · PackNet-SAN
- **Video depth (6):** Video Depth Anything · Online VDA · DA3-Streaming · DepthCrafter · ChronoDepth · AsyncMDE
- **Self-sup + scale (15):** SfMLearner · Monodepth · Monodepth2 · SC-Depth · PackNet-SfM · ManyDepth · TRI multi-frame · DynamicDepth · MonoViT · Lite-Mono · MonoDEVSNet · DNet · CameraHeight'23 · MonoPP · Gamma-from-Mono (+ VI-rescaling 2509.08159)
- **Stereo (14):** PSMNet · GwcNet · GA-Net · AANet · HITNet · RAFT-Stereo · CREStereo · ACVNet · IGEV · MobileStereoNet · StereoAnything · FoundationStereo · Fast-FoundationStereo · camera comparison 2501.07421
- **Depth completion (11):** Uhrig'17 · IP-Basic · DepthNormal · NLSPN · PENet · EfficientPENet · CompletionFormer · SemAttNet · planar-LiDAR fusion · CU-Net · Prompt Depth Anything
- **Per-object distance & mono 3D (12):** DisNet · Dist-YOLO · Zhu&Fang · railway bbox-distance · indoor AV distance · ADAS distance · Pseudo-LiDAR · SMOKE · FCOS3D · DETR3D · BEVDet · BEVFormer
- **Speed estimation (30+):** Kampelmühler · TuSimple bench · Song ICRA20 · QD-3DT · CenterTrack · CenterFusion · CFTrack · Full-Velocity radar · DeepCrashTest · 3DCMA · ego-flow'25 · RNN/Transformer speed · KITTI speed eval 1907.06989 · FlowNet/FlowNet2/PWC-Net/RAFT · Dubská'14 · Sochor calib · BrnoCompSpeed · Luvizon · Revaud ICCV21 · ISPRS'20 · AI City'18 (+TeamUW) · VP calibration 2103.11438 · Efficient speed 2505.01203 · YOLOv8 speed 2406.07710 · YOLOv5s+DeepSORT Frontiers'24 · deep-homography'24 · line-homography ETASR'25 · SciReports'25 · UAV Algorithms'24 · view-invariant sim 2206.00343 · audio-video 2212.01651 · forensic occlusion MTAP'26
- **TTC & VRU safety (5):** TTC forecasting 1903.09102 · Binary TTC · constant-jerk TTC · collision review · cyclist orientation 2004.11909
- **Edge deployment (8):** FastDepth · RT-MonoDepth · GuidedDecoding · embedded human depth · ZipDepth · AsyncMDE · DA3 ROS2/TensorRT · Fast-FoundationStereo
- **Datasets (30+):** KITTI (+DC, +360) · DDAD · DrivingStereo · nuScenes · Waymo · Argoverse 2 · ApolloScape · A2D2 · PandaSet · ONCE · Lyft L5 · Cityscapes · Mapillary PSD · ROVR · DENSE/STF · RobotCar · DIODE · Make3D · UASOL · SeasonDepth · vKITTI2 · SYNTHIA · CARLA · SHIFT · SceneFlow · TartanAir · Hypersim · Sintel · NYUv2 · ScanNet · SUN RGB-D · MegaDepth · BrnoCompSpeed · AI City · TuSimple velocity · Luvizon · comma2k19 · speedchallenge · VS13
- **Context:** IDM 25-year survey (arXiv:2506.05909 — the repo's example paper; longitudinal control consumer of d & Δv)
