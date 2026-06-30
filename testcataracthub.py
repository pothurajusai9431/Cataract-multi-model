"""
test_cataract_hub.py  ── 20 Tests (Unit + Integration)
=======================================================
A complete test suite for the Cataract Hub Flask application covering:
  • Unit tests  — helper/utility functions in app.py
  • Integration tests — Flask routes via the test client

HOW TO RUN
----------
    python test_cataract_hub.py                    # built-in unittest (always works)
    python -m pytest test_cataract_hub.py -v       # pytest (if installed)

COVERAGE MAP
============
──────────────────────────────── INTEGRATION TESTS ────────────────────────────────
TC-01  GET  /          → HTTP 200 OK, page renders without error
TC-02  GET  /          → Upload form, drop-zone, and brand name present in HTML
TC-03  POST /          → No file attached → error message shown
TC-04  POST /          → Wrong file type (.txt) → unsupported-type error
TC-05  POST /          → Image too small (30x30) → Layer 1A size error
TC-06  POST /          → Blank solid image (sigma=0) → Layer 1B blank error
TC-07  POST /          → Random-noise image (lap_var~49k) → Layer 2 screenshot error
TC-08  POST /          → Pure-colour cartoon (hi_sat=1.0) → Layer 2b cartoon error
TC-09  POST /          → White-background image (96% white) → Layer 2c background error
TC-10  POST /chat      → JSON response always contains "reply" key

─────────────────────────────────── UNIT TESTS ─────────────────────────────────────
TC-11  allowed_file()  → accepts all 7 whitelisted image extensions
TC-12  allowed_file()  → rejects non-image extensions (.exe .php .pdf .py)
TC-13  allowed_file()  → handles filenames with NO extension
TC-14  _near_border()  → returns True for corner/edge points
TC-15  _near_border()  → returns False for center points
TC-16  _group_score()  → score = len(group) x max_detection_size
TC-17  _group_center() → centroid is the arithmetic mean of cx/cy values
TC-18  _filter_small_dets() → removes detections < 40% of largest
TC-19  _group_dets()   → merges close detections, keeps far ones separate
TC-20  _circle_interior_mean() → returns correct mean brightness inside a circle
"""

import io
import sys
import os
import json
import unittest

import cv2
import numpy as np

# ── resolve app root ─────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import (                        # noqa: E402
    app,
    allowed_file,
    _near_border,
    _group_score,
    _group_center,
    _filter_small_dets,
    _group_dets,
    _circle_interior_mean,
)


# ═══════════════════════════════════════════════════════════════
#  SYNTHETIC IMAGE FACTORIES
# ═══════════════════════════════════════════════════════════════

def _encode(arr: np.ndarray, fmt: str = ".jpg") -> bytes:
    """Encode a numpy BGR array to image bytes for multipart upload."""
    ok, buf = cv2.imencode(fmt, arr)
    assert ok, f"cv2.imencode failed for format {fmt}"
    return buf.tobytes()


def _img_tiny() -> bytes:
    """30x30 solid-grey — triggers Layer 1A (dimensions < 50 px)."""
    return _encode(np.full((30, 30, 3), 110, dtype=np.uint8))


def _img_blank() -> bytes:
    """200x200 perfectly uniform grey — triggers Layer 1B (sigma = 0)."""
    return _encode(np.full((200, 200, 3), 128, dtype=np.uint8))


def _img_noise() -> bytes:
    """200x200 random pixels — triggers Layer 2 (lap_var ~49000 > 8000)."""
    rng = np.random.default_rng(seed=42)
    return _encode(rng.integers(0, 256, (200, 200, 3), dtype=np.uint8))


def _img_cartoon() -> bytes:
    """
    Three stripes of pure primary colours.
    hi_sat_frac=1.0 > 0.75, skin_frac=0.0 < 0.08 -- triggers Layer 2b Rule A.
    """
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[  0: 67, :] = [255, 0,   0  ]
    img[ 67:134, :] = [0,   255, 0  ]
    img[134:200, :] = [0,   0,   255]
    return _encode(img)


def _img_white_bg() -> bytes:
    """
    ~96% of pixels R,G,B > 235 — triggers Layer 2c (white_frac=0.96 > 0.35).
    Represents a hand/document photographed on a light-box.
    """
    img = np.full((200, 200, 3), 245, dtype=np.uint8)
    img[80:120, 80:120] = [100, 130, 170]   # small patch keeps sigma > 5
    return _encode(img)


def _post_image(client, image_bytes: bytes,
                filename: str = "eye.jpg",
                content_type: str = "image/jpeg"):
    """POST multipart/form-data to / with the given image bytes."""
    return client.post(
        "/",
        data={"file": (io.BytesIO(image_bytes), filename, content_type)},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION TESTS  (TC-01 to TC-10)
# ═══════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):
    """
    Route-level tests using Flask's built-in test client.
    No real server or real eye images needed.
    """

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"]          = True
        app.config["WTF_CSRF_ENABLED"] = False
        cls.client = app.test_client()

    # ------------------------------------------------------------------ TC-01
    def test_01_homepage_returns_200(self):
        """
        GET / must return HTTP 200.
        Confirms Flask routing is wired and the Jinja template renders
        without any runtime error.
        """
        resp = self.client.get("/")
        self.assertEqual(
            resp.status_code, 200,
            f"Expected HTTP 200 on GET /, got {resp.status_code}"
        )

    # ------------------------------------------------------------------ TC-02
    def test_02_homepage_html_contains_key_elements(self):
        """
        The rendered HTML must contain three landmarks:
          <form    -- the multipart upload form tag
          drop-zone -- the drag-and-drop area id
          Cataract  -- the application brand name
        Confirms that Jinja template variables are injected correctly and
        the static layout has not been accidentally removed.
        """
        html = self.client.get("/").data.decode("utf-8", errors="replace")
        self.assertIn("<form",     html, "Upload <form> tag missing from homepage")
        self.assertIn("drop-zone", html, "id='drop-zone' missing from homepage")
        self.assertIn("Cataract",  html, "Brand name 'Cataract' missing from homepage")

    # ------------------------------------------------------------------ TC-03
    def test_03_post_no_file_returns_error_message(self):
        """
        Submitting the upload form with NO file must:
          * Return HTTP 200  (page re-renders, no redirect loop)
          * Show a user-friendly error (keyword: 'no file', 'select', or 'error')
        Targets the first guard clause at the top of the POST handler.
        """
        resp = self.client.post(
            "/", data={},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="replace").lower()
        self.assertTrue(
            "no file" in html or "select" in html or "error" in html,
            "Expected a 'no file selected' message when no file is submitted"
        )

    # ------------------------------------------------------------------ TC-04
    def test_04_wrong_extension_txt_is_rejected(self):
        """
        Uploading a .txt file must be caught by allowed_file() before any
        OpenCV or model code runs.
        Expected error keyword: 'unsupported' or 'file type'.
        """
        resp = self.client.post(
            "/",
            data={"file": (io.BytesIO(b"not an image"), "notes.txt", "text/plain")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="replace").lower()
        self.assertTrue(
            "unsupported" in html or "file type" in html or "error" in html,
            "Expected unsupported-file-type error for a .txt upload"
        )

    # ------------------------------------------------------------------ TC-05
    def test_05_too_small_image_shows_size_error(self):
        """
        A 30x30 JPEG (both dimensions below the 50 px minimum) must be
        rejected by Layer 1A immediately after cv2.imread().
        Expected keyword in rendered error HTML: 'small'.
        """
        resp = _post_image(self.client, _img_tiny())
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="replace").lower()
        self.assertIn("small", html,
            "Expected 'small' in the error message for a 30x30 image")

    # ------------------------------------------------------------------ TC-06
    def test_06_blank_image_shows_blank_error(self):
        """
        A perfectly uniform 200x200 grey image (pixel std-dev = 0) must be
        rejected by Layer 1B.
        Expected keywords in HTML: 'blank' or 'solid'.
        """
        resp = _post_image(self.client, _img_blank())
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="replace").lower()
        self.assertTrue(
            "blank" in html or "solid" in html,
            "Expected 'blank' or 'solid' in the error for a uniform-colour image"
        )

    # ------------------------------------------------------------------ TC-07
    def test_07_noise_image_shows_screenshot_error(self):
        """
        A fully random pixel image has Laplacian variance ~49 000, far above
        MAX_LAP_VARIANCE (8 000). Layer 2 must catch it.
        Expected keywords: 'screenshot' or 'computer'.
        """
        resp = _post_image(self.client, _img_noise())
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="replace").lower()
        self.assertTrue(
            "screenshot" in html or "computer" in html,
            "Expected 'screenshot'/'computer' error for a random-noise image"
        )

    # ------------------------------------------------------------------ TC-08
    def test_08_cartoon_image_shows_cartoon_error(self):
        """
        Three stripes of pure primary colours produce hi_sat_frac=1.0 and
        skin_frac=0.0 -- satisfying Rule A in Layer 2b.
        Expected keywords: 'cartoon' or 'illustration'.
        """
        resp = _post_image(self.client, _img_cartoon())
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="replace").lower()
        self.assertTrue(
            "cartoon" in html or "illustration" in html,
            "Expected 'cartoon'/'illustration' error for a pure-colour image"
        )

    # ------------------------------------------------------------------ TC-09
    def test_09_white_background_image_shows_background_error(self):
        """
        An image where ~96% of pixels have R,G,B > 235 simulates a
        hand or document on a light-box. white_frac=0.96 >> threshold 0.35.
        Layer 2c must reject it.
        Expected keywords: 'background', 'hand', or 'object'.
        """
        resp = _post_image(self.client, _img_white_bg())
        self.assertEqual(resp.status_code, 200)
        html = resp.data.decode("utf-8", errors="replace").lower()
        self.assertTrue(
            "background" in html or "hand" in html or "object" in html,
            "Expected background/hand/object error for white-background image"
        )

    # ------------------------------------------------------------------ TC-10
    def test_10_chat_endpoint_returns_json_reply_key(self):
        """
        POST /chat with a JSON body must always return:
          * HTTP 200
          * Content-Type: application/json
          * A JSON object with a non-empty 'reply' string

        This holds whether or not GROQ_API_KEY is set -- the route wraps all
        exceptions and returns them as a JSON reply, so the key is guaranteed.
        """
        payload = json.dumps({"message": "What is a cataract?", "language": "English"})
        resp = self.client.post("/chat", data=payload,
                                content_type="application/json")
        self.assertEqual(resp.status_code, 200,
            f"Expected HTTP 200 from /chat, got {resp.status_code}")

        try:
            data = json.loads(resp.data)
        except json.JSONDecodeError as exc:
            self.fail(f"/chat did not return valid JSON: {exc}")

        self.assertIn("reply", data,
            f"Expected 'reply' key in /chat JSON, got keys: {list(data.keys())}")
        self.assertIsInstance(data["reply"], str,
            f"'reply' must be a string, got {type(data['reply'])}")
        self.assertGreater(len(data["reply"]), 0,
            "Expected a non-empty 'reply' string from /chat")


# ═══════════════════════════════════════════════════════════════
#  UNIT TESTS  (TC-11 to TC-20)
# ═══════════════════════════════════════════════════════════════

class TestUnitHelpers(unittest.TestCase):
    """
    Pure unit tests for each utility function in app.py.
    No HTTP, no Flask client, no model loading required.
    Each test is deterministic and independent.
    """

    # ------------------------------------------------------------------ TC-11
    def test_11_allowed_file_accepts_all_valid_extensions(self):
        """
        allowed_file() must return True for every extension in the whitelist:
        jpg, jpeg, png, webp, bmp, gif, tiff -- and must be case-insensitive
        so that 'EYE.JPG' and 'scan.PNG' are also accepted.

        This prevents users from being blocked by their OS capitalising the
        extension automatically (common on Windows).
        """
        valid_names = [
            "eye.jpg",   "scan.jpeg",  "retina.png",  "fundus.webp",
            "iris.bmp",  "test.gif",   "image.tiff",
            # uppercase variants
            "EYE.JPG",   "SCAN.PNG",   "photo.WEBP",  "img.TIFF",
        ]
        for name in valid_names:
            with self.subTest(filename=name):
                self.assertTrue(
                    allowed_file(name),
                    f"allowed_file('{name}') should return True for a valid extension"
                )

    # ------------------------------------------------------------------ TC-12
    def test_12_allowed_file_rejects_non_image_extensions(self):
        """
        allowed_file() must return False for extensions outside the whitelist.
        Prevents arbitrary file uploads: executables, scripts, PDFs, archives.
        Each subtest is independent -- one failure does not skip others.
        """
        invalid_names = [
            "virus.exe",  "shell.php",  "report.pdf",
            "script.py",  "notes.txt",  "archive.zip",  "data.csv",
        ]
        for name in invalid_names:
            with self.subTest(filename=name):
                self.assertFalse(
                    allowed_file(name),
                    f"allowed_file('{name}') should return False"
                )

    # ------------------------------------------------------------------ TC-13
    def test_13_allowed_file_rejects_filename_with_no_dot(self):
        """
        A filename containing no '.' at all must return False.
        The function checks for '.' before splitting -- this is the first
        guard and prevents index errors on rsplit('.', 1).
        """
        no_ext_names = ["noextension", "eyescan", "README", "uploadedfile"]
        for name in no_ext_names:
            with self.subTest(filename=name):
                self.assertFalse(
                    allowed_file(name),
                    f"allowed_file('{name}') should return False (no extension)"
                )

    # ------------------------------------------------------------------ TC-14
    def test_14_near_border_returns_true_for_edge_and_corner_points(self):
        """
        _near_border(cx, cy, w, h, frac=0.17) must return True when the
        point lies within the outer 17% margin of the image.

        Points tested: all four corners and all four edge midpoints.
        These correspond to slit-lamp illumination arcs that appear in
        the corner of clinical photographs and must not be counted as eyes.
        """
        w, h = 100, 100
        # frac=0.17 means the border zone is 0..16 and 84..100 in each axis
        edge_points = [
            (5,  5,  "top-left corner"),
            (95, 5,  "top-right corner"),
            (5,  95, "bottom-left corner"),
            (95, 95, "bottom-right corner"),
            (50, 5,  "top edge midpoint"),
            (50, 95, "bottom edge midpoint"),
            (5,  50, "left edge midpoint"),
            (95, 50, "right edge midpoint"),
        ]
        for cx, cy, label in edge_points:
            with self.subTest(point=label):
                self.assertTrue(
                    _near_border(cx, cy, w, h),
                    f"Point ({cx},{cy}) '{label}' should be flagged as near-border"
                )

    # ------------------------------------------------------------------ TC-15
    def test_15_near_border_returns_false_for_interior_points(self):
        """
        _near_border() must return False for points well inside the image
        (farther than 17% from any edge).

        Interior detections represent the iris and pupil -- they must never
        be suppressed by the near-border filter.
        """
        w, h = 100, 100
        interior_points = [
            (50, 50, "dead center"),
            (30, 30, "upper-left interior"),
            (70, 70, "lower-right interior"),
            (50, 40, "slightly above center"),
            (25, 25, "just inside the 17% zone"),
        ]
        for cx, cy, label in interior_points:
            with self.subTest(point=label):
                self.assertFalse(
                    _near_border(cx, cy, w, h),
                    f"Point ({cx},{cy}) '{label}' should NOT be near-border"
                )

    # ------------------------------------------------------------------ TC-16
    def test_16_group_score_equals_count_times_max_size(self):
        """
        _group_score(g) returns  len(g) x max(d[2] for d in g).

        This formula balances two factors:
          * len(g)       -- how many detector hits agree (vote count)
          * max size     -- how large the dominant detection is
        The product is used to rank competing iris candidates so that the
        strongest group is always selected first.

        Three deterministic cases are verified:
          A. Single detection of size 100    -> 1 x 100 = 100
          B. Two detections, sizes 80 + 100  -> 2 x 100 = 200
          C. Three equal detections of size 30 -> 3 x 30 = 90
        """
        # A
        self.assertEqual(_group_score([(50, 50, 100)]), 100,
            "TC-16A: 1 x 100 = 100")
        # B
        self.assertEqual(
            _group_score([(50, 50, 80), (60, 60, 100)]), 200,
            "TC-16B: 2 x max(80,100) = 200"
        )
        # C
        g3 = [(10, 10, 30), (20, 20, 30), (30, 30, 30)]
        self.assertEqual(_group_score(g3), 90,
            "TC-16C: 3 x 30 = 90")

    # ------------------------------------------------------------------ TC-17
    def test_17_group_center_returns_arithmetic_mean_of_all_points(self):
        """
        _group_center(g) must return the arithmetic mean of all (cx, cy) pairs.
        The returned centroid is used to measure separation between competing
        iris groups and decide whether two groups represent two eyes or one.

        Three cases:
          A. Two diagonal corners  -> centroid (50, 50)
          B. Single point          -> centroid equals the point itself
          C. Four symmetric points -> centroid (0, 0)
        """
        # A: Opposite corners
        g_a = [(0, 0, 10), (100, 100, 10)]
        cx_a, cy_a = _group_center(g_a)
        self.assertAlmostEqual(cx_a, 50.0, places=3,
            msg="TC-17A: center-x of diagonal pair should be 50.0")
        self.assertAlmostEqual(cy_a, 50.0, places=3,
            msg="TC-17A: center-y of diagonal pair should be 50.0")

        # B: Single-point group
        g_b = [(73, 41, 20)]
        cx_b, cy_b = _group_center(g_b)
        self.assertEqual(cx_b, 73.0, "TC-17B: single-point center-x = 73")
        self.assertEqual(cy_b, 41.0, "TC-17B: single-point center-y = 41")

        # C: Four symmetric points
        g_c = [(-10, -10, 5), (10, -10, 5), (-10, 10, 5), (10, 10, 5)]
        cx_c, cy_c = _group_center(g_c)
        self.assertAlmostEqual(cx_c, 0.0, places=5,
            msg="TC-17C: symmetric group center-x should be 0.0")
        self.assertAlmostEqual(cy_c, 0.0, places=5,
            msg="TC-17C: symmetric group center-y should be 0.0")

    # ------------------------------------------------------------------ TC-18
    def test_18_filter_small_dets_drops_detections_below_40_percent(self):
        """
        _filter_small_dets(dets, min_size_ratio=0.40) removes any detection
        whose size is less than 40% of the largest detection.

        Purpose: eliminate eyelash rows, eyelid-crease hits, and slit-lamp
        arc artifacts that the Haar cascade occasionally fires on.

        Input sizes: [100, 50, 20]
          threshold = 100 x 0.40 = 40
          100 >= 40 -> kept
           50 >= 40 -> kept
           20 <  40 -> DROPPED
        Expected output sizes (sorted desc): [100, 50]

        Also verifies that an empty input returns empty output (no crash).
        """
        dets = [(10, 10, 100), (20, 20, 50), (30, 30, 20)]
        result = _filter_small_dets(dets, min_size_ratio=0.40)
        sizes = sorted([d[2] for d in result], reverse=True)

        self.assertEqual(sizes, [100, 50],
            f"Expected kept sizes [100, 50], got {sizes}")
        self.assertNotIn(20, [d[2] for d in result],
            "Size-20 detection should have been removed (< 40% of max=100)")

        # Edge case: empty list
        self.assertEqual(_filter_small_dets([]), [],
            "Empty input must return empty list without raising")

    # ------------------------------------------------------------------ TC-19
    def test_19_group_dets_merges_close_and_separates_distant(self):
        """
        _group_dets(dets, prox) must:
          * Merge detections within `prox` pixels of their running centroid
          * Keep detections that are farther apart in separate groups
          * Never lose any detection (total items preserved)

        Three sub-tests:
          A. Two close detections (dist ~7 px, prox=20)   -> 1 group
          B. Two far  detections  (dist 283 px, prox=30)  -> 2 groups
          C. Two close + one far                           -> 2 groups, 3 items total
        """
        # A: Close pair merges
        close = [(50, 50, 30), (55, 55, 28)]
        groups_a = _group_dets(close, prox=20)
        self.assertEqual(len(groups_a), 1,
            f"TC-19A: Two close dets (dist~7px) should merge to 1 group, got {len(groups_a)}")

        # B: Far pair stays separate
        far = [(10, 10, 20), (210, 210, 20)]
        groups_b = _group_dets(far, prox=30)
        self.assertEqual(len(groups_b), 2,
            f"TC-19B: Two distant dets should remain 2 groups, got {len(groups_b)}")

        # C: Mixed -- close pair + isolated far point
        mixed = [(50, 50, 30), (55, 55, 28), (200, 200, 30)]
        groups_c = _group_dets(mixed, prox=20)
        self.assertEqual(len(groups_c), 2,
            f"TC-19C: Close pair + far det should give 2 groups, got {len(groups_c)}")
        total = sum(len(g) for g in groups_c)
        self.assertEqual(total, 3,
            f"TC-19C: Total dets across groups must be 3 (no data loss), got {total}")

    # ------------------------------------------------------------------ TC-20
    def test_20_circle_interior_mean_computes_correct_pixel_brightness(self):
        """
        _circle_interior_mean(gray, cx, cy, radius) returns the mean pixel
        brightness of all pixels inside the circular region.

        This is the anatomy gate (Layer 5b) for Hough-detected circles:
          * Real iris tissue:   mean <= 185 -> PASS (accepted as possible eye)
          * Palm / bright skin: mean >  185 -> FAIL (rejected as non-eye)

        Four sub-tests:
          A. Uniform dark  image (all=60)  -> mean ~60,  passes gate (<185)
          B. Uniform bright image (all=220) -> mean ~220, fails  gate (>185)
          C. Dark iris patch on bright background -> mean ~60, passes gate
          D. Circle centred right at image corner (partial overlap) ->
             result is a valid float in range [0, 255] (no crash, no NaN)

        NOTE on radius=0:
          The mask  (X-cx)^2 + (Y-cy)^2 <= 0  is True for exactly ONE pixel
          (the center pixel itself, distance^2 = 0).  So pixels.size = 1 and
          the function returns that pixel's brightness value -- NOT 255.0.
          The 255.0 fallback only fires when pixels.size == 0, which cannot
          happen for radius=0 when (cx, cy) is inside the image.
        """
        # A: All-dark -- iris-like
        gray_dark = np.full((200, 200), 60, dtype=np.uint8)
        mean_a = _circle_interior_mean(gray_dark, 100, 100, 50)
        self.assertAlmostEqual(mean_a, 60.0, delta=0.5,
            msg=f"TC-20A: Expected mean~60 for dark image, got {mean_a:.2f}")
        self.assertLess(mean_a, 185,
            "TC-20A: Dark circle should pass anatomy gate (mean < 185)")

        # B: All-bright -- hand/palm-like
        gray_bright = np.full((200, 200), 220, dtype=np.uint8)
        mean_b = _circle_interior_mean(gray_bright, 100, 100, 50)
        self.assertAlmostEqual(mean_b, 220.0, delta=0.5,
            msg=f"TC-20B: Expected mean~220 for bright image, got {mean_b:.2f}")
        self.assertGreater(mean_b, 185,
            "TC-20B: Bright circle should fail anatomy gate (mean > 185)")

        # C: Realistic eye-like (dark iris patch on bright background)
        eye_like = np.full((200, 200), 200, dtype=np.uint8)
        cv2.circle(eye_like, (100, 100), 40, 60, -1)   # dark iris
        mean_c = _circle_interior_mean(eye_like, 100, 100, 40)
        self.assertLess(mean_c, 185,
            f"TC-20C: Eye-like dark iris should pass anatomy gate, mean={mean_c:.2f}")

        # D: Circle centred at the top-left corner -- most of the circle lies
        #    outside the image, so only a small arc of pixels is captured.
        #    The function must still return a finite float in [0, 255] without
        #    crashing or producing NaN / Inf.
        #    (This exercises the np.ogrid boundary behaviour used in the mask.)
        gray_corner = np.full((200, 200), 130, dtype=np.uint8)
        mean_d = _circle_interior_mean(gray_corner, 0, 0, 30)
        self.assertIsInstance(mean_d, float,
            f"TC-20D: Expected a float for corner-circle, got {type(mean_d)}")
        self.assertGreaterEqual(mean_d, 0.0,
            f"TC-20D: Mean must be >= 0, got {mean_d}")
        self.assertLessEqual(mean_d, 255.0,
            f"TC-20D: Mean must be <= 255, got {mean_d}")


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  CATARACT HUB -- Full Test Suite (20 tests)")
    print("  Integration Tests (TC-01 to TC-10)")
    print("  Unit Tests        (TC-11 to TC-20)")
    print("=" * 65)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestUnitHelpers))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 65)
    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"  RESULT: {passed}/{total} tests passed")
    if result.wasSuccessful():
        print("  ALL TESTS PASSED -- safe to demo in front of panel members!")
    else:
        print("  Some tests failed. Check the details above.")
    print("=" * 65)

    sys.exit(0 if result.wasSuccessful() else 1)