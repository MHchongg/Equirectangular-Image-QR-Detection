import os
import cv2
import numpy as np
import py360convert


# =========================================================
# Configuration
# =========================================================

INPUT_IMAGE = "equirectangular with QR.jpeg"

CUBEMAP_OUTPUT_DIR = "cubemap"
PERSPECTIVE_OUTPUT_DIR = "perspective"
UPSCALE_OUTPUT_DIR = "upscale"

# Higher resolution preserves more detail from the panorama.
CUBEMAP_FACE_SIZE = 2048

# Perspective output resolution.
PERSPECTIVE_OUTPUT_SIZE = (2048, 2048)

# Perspective field of view.
# 90° gives a reasonably wide view while keeping distortion manageable.
FOV = (90, 90)

# Horizontal rotation angles.
YAW_ANGLES = range(0, 360, 45)

# Vertical rotation angles.
# Negative = look down
# Positive = look up
PITCH_ANGLES = [-30, 0, 30]

# Upscaling factor.
UPSCALE_FACTOR = 2


# =========================================================
# Utility
# =========================================================

def ensure_directories():
    """
    Create all output directories.
    """

    os.makedirs(CUBEMAP_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PERSPECTIVE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(UPSCALE_OUTPUT_DIR, exist_ok=True)


# =========================================================
# Equirectangular -> Cubemap
# =========================================================

def convert_to_cubemap(image):
    """
    Convert an equirectangular image into six cubemap faces.

    Returns:
        Dictionary containing:
        F, R, B, L, U, D
    """

    print("\nConverting equirectangular -> cubemap...")

    faces = py360convert.e2c(
        image,
        face_w=CUBEMAP_FACE_SIZE,
        mode="bilinear",
        cube_format="dict",
    )

    return faces


def save_cubemap_faces(faces):
    """
    Save all cubemap faces.
    """

    face_names = ["F", "R", "B", "L", "U", "D"]

    paths = {}

    for face_name in face_names:

        face = faces[face_name]

        output_path = os.path.join(
            CUBEMAP_OUTPUT_DIR,
            f"{face_name}.jpg"
        )

        cv2.imwrite(output_path, face)

        paths[face_name] = output_path

        print(f"  Cubemap saved: {output_path}")

    return paths


# =========================================================
# Equirectangular -> Perspective Views
# =========================================================

def convert_to_perspective_views(image):
    """
    Generate overlapping perspective views.

    8 horizontal angles:
        0, 45, 90, ..., 315

    3 vertical angles:
        -30, 0, +30

    Total:
        8 x 3 = 24 views

    Returns:
        Dictionary:
        {
            "yaw_000_pitch_-30": image,
            ...
        }
    """

    print("\nGenerating perspective views...")

    views = {}

    for yaw in YAW_ANGLES:

        for pitch in PITCH_ANGLES:

            print(
                f"  Generating "
                f"yaw={yaw:3d}°, "
                f"pitch={pitch:+3d}°"
            )

            view = py360convert.e2p(
                image,
                fov_deg=FOV,
                u_deg=yaw,
                v_deg=pitch,
                out_hw=PERSPECTIVE_OUTPUT_SIZE,
                mode="bilinear",
            )

            view_name = (
                f"yaw_{yaw:03d}_pitch_{pitch:+03d}"
            )

            views[view_name] = view

    return views


def save_perspective_views(views):
    """
    Save all perspective views.
    """

    paths = {}

    for view_name, image in views.items():

        output_path = os.path.join(
            PERSPECTIVE_OUTPUT_DIR,
            f"{view_name}.jpg"
        )

        cv2.imwrite(output_path, image)

        paths[view_name] = output_path

        print(
            f"  Perspective saved: "
            f"{output_path}"
        )

    return paths


# =========================================================
# Image Preprocessing
# =========================================================

def upscale_image(image, interpolation):
    """
    Upscale an image.

    Returns:
        Upscaled image.
    """

    height, width = image.shape[:2]

    new_width = width * UPSCALE_FACTOR
    new_height = height * UPSCALE_FACTOR

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interpolation
    )


def create_image_variants(image):
    """
    Create several versions of an image.

    The goal is to give the QR detector multiple
    representations to work with.

    Returns:
        Dictionary containing image variants.
    """

    variants = {}

    # -----------------------------------------------------
    # 1. Original
    # -----------------------------------------------------

    variants["original"] = image

    # -----------------------------------------------------
    # 2. Grayscale
    # -----------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    variants["gray"] = gray

    # -----------------------------------------------------
    # 3. CLAHE
    #
    # Improve local contrast.
    # -----------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_image = clahe.apply(gray)

    variants["clahe"] = clahe_image

    # -----------------------------------------------------
    # 4. Otsu threshold
    # -----------------------------------------------------

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants["otsu"] = otsu

    # -----------------------------------------------------
    # 5. Adaptive threshold
    # -----------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    variants["adaptive"] = adaptive

    return variants


# =========================================================
# QR Detection
# =========================================================

def scan_qr(image, image_name):
    """
    Detect and decode QR codes from an image.

    Returns:
        List of detected QR information.
    """

    detector = cv2.QRCodeDetector()

    results = []

    # =====================================================
    # Try multi QR detection
    # =====================================================

    try:

        success, decoded_info, points, _ = (
            detector.detectAndDecodeMulti(image)
        )

        if success and decoded_info:

            for i, data in enumerate(decoded_info):

                if not data:
                    continue

                qr_info = {
                    "data": data,
                    "points": None,
                    "center": None,
                    "source": image_name,
                }

                if points is not None:

                    qr_points = points[i]

                    center_x = float(
                        np.mean(qr_points[:, 0])
                    )

                    center_y = float(
                        np.mean(qr_points[:, 1])
                    )

                    qr_info["points"] = qr_points
                    qr_info["center"] = (
                        center_x,
                        center_y,
                    )

                results.append(qr_info)

    except cv2.error:
        pass

    # =====================================================
    # Fallback to single QR detection
    # =====================================================

    if not results:

        try:

            data, points, _ = (
                detector.detectAndDecode(image)
            )

            if data:

                qr_info = {
                    "data": data,
                    "points": None,
                    "center": None,
                    "source": image_name,
                }

                if points is not None:

                    qr_points = points[0]

                    center_x = float(
                        np.mean(qr_points[:, 0])
                    )

                    center_y = float(
                        np.mean(qr_points[:, 1])
                    )

                    qr_info["points"] = qr_points
                    qr_info["center"] = (
                        center_x,
                        center_y,
                    )

                results.append(qr_info)

        except cv2.error:
            pass

    return results


# =========================================================
# Scan One Image Using Multiple Preprocessing Methods
# =========================================================

def scan_image_with_variants(
    image,
    image_name,
    global_results
):
    """
    Upscale and scan one image using several
    preprocessing methods and interpolation methods.
    """

    # =====================================================
    # Try two different upscaling algorithms
    # =====================================================

    interpolation_methods = {
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    for interpolation_name, interpolation in (
        interpolation_methods.items()
    ):

        print(
            f"\n[{image_name}] "
            f"Upscaling using {interpolation_name}..."
        )

        upscaled = upscale_image(
            image,
            interpolation
        )

        # -------------------------------------------------
        # Create preprocessing variants
        # -------------------------------------------------

        variants = create_image_variants(
            upscaled
        )

        for variant_name, variant in variants.items():

            scan_name = (
                f"{image_name}/"
                f"{interpolation_name}/"
                f"{variant_name}"
            )

            print(
                f"  Scanning: {scan_name}"
            )

            results = scan_qr(
                variant,
                scan_name
            )

            # ---------------------------------------------
            # Store results
            # ---------------------------------------------

            for result in results:

                data = result["data"]

                if data not in global_results:

                    global_results[data] = {
                        "data": data,
                        "detections": []
                    }

                global_results[data][
                    "detections"
                ].append(result)

                print(
                    f"    QR FOUND: {data}"
                )

                if result["center"]:

                    x, y = result["center"]

                    print(
                        f"    Center: "
                        f"({x:.1f}, {y:.1f})"
                    )


# =========================================================
# Scan Cubemap
# =========================================================

def scan_cubemap_faces(
    faces,
    global_results
):
    """
    Scan all six cubemap faces.
    """

    print(
        "\n========================================"
    )

    print(
        "Scanning cubemap faces"
    )

    print(
        "========================================"
    )

    face_names = [
        "F",
        "R",
        "B",
        "L",
        "U",
        "D",
    ]

    for face_name in face_names:

        print(
            f"\n----------------------------------------"
        )

        print(
            f"Cubemap face: {face_name}"
        )

        print(
            f"----------------------------------------"
        )

        face = faces[face_name]

        scan_image_with_variants(
            face,
            f"cubemap_{face_name}",
            global_results
        )


# =========================================================
# Scan Perspective Views
# =========================================================

def scan_perspective_views(
    views,
    global_results
):
    """
    Scan all perspective views.
    """

    print(
        "\n========================================"
    )

    print(
        "Scanning perspective views"
    )

    print(
        "========================================"
    )

    for view_name, image in views.items():

        print(
            f"\n----------------------------------------"
        )

        print(
            f"Perspective view: {view_name}"
        )

        print(
            f"----------------------------------------"
        )

        scan_image_with_variants(
            image,
            view_name,
            global_results
        )


# =========================================================
# Print Final Results
# =========================================================

def print_results(global_results):
    """
    Print unique QR results.
    """

    print(
        "\n\n========================================"
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "========================================"
    )

    if not global_results:

        print(
            "\nNo QR codes were successfully decoded."
        )

        return

    print(
        f"\nUnique QR codes found: "
        f"{len(global_results)}"
    )

    for index, (data, info) in enumerate(
        global_results.items(),
        start=1
    ):

        print(
            f"\nQR #{index}"
        )

        print(
            f"  Data: {data}"
        )

        print(
            f"  Successful detections: "
            f"{len(info['detections'])}"
        )

        # Print where this QR was found.
        printed_sources = set()

        for detection in info["detections"]:

            source = detection["source"]

            if source in printed_sources:
                continue

            printed_sources.add(source)

            center = detection["center"]

            if center:

                x, y = center

                print(
                    f"    - {source}"
                    f" at ({x:.1f}, {y:.1f})"
                )

            else:

                print(
                    f"    - {source}"
                )


# =========================================================
# Main
# =========================================================

def main():

    # =====================================================
    # 1. Check input image
    # =====================================================

    if not os.path.exists(INPUT_IMAGE):

        print(
            f"ERROR: Cannot find "
            f"{INPUT_IMAGE}"
        )

        return

    # =====================================================
    # 2. Create directories
    # =====================================================

    ensure_directories()

    # =====================================================
    # 3. Load image
    # =====================================================

    image = cv2.imread(INPUT_IMAGE)

    if image is None:

        print(
            f"ERROR: Failed to load "
            f"{INPUT_IMAGE}"
        )

        return

    # =====================================================
    # 4. Display panorama information
    # =====================================================

    height, width = image.shape[:2]

    print(
        "\n========================================"
    )

    print(
        "INPUT IMAGE"
    )

    print(
        "========================================"
    )

    print(
        f"Width : {width}"
    )

    print(
        f"Height: {height}"
    )

    print(
        f"Ratio : {width / height:.2f}"
    )

    if abs(
        (width / height) - 2.0
    ) > 0.1:

        print(
            "\nWARNING:"
        )

        print(
            "Image is not approximately 2:1."
        )

        print(
            "Make sure this is an "
            "equirectangular panorama."
        )

    # =====================================================
    # Store unique QR results
    # =====================================================

    global_results = {}

    # =====================================================
    # 5. Cubemap
    # =====================================================

    faces = convert_to_cubemap(
        image
    )

    save_cubemap_faces(
        faces
    )

    # =====================================================
    # 6. Scan cubemap
    # =====================================================

    scan_cubemap_faces(
        faces,
        global_results
    )

    # =====================================================
    # 7. Perspective views
    # =====================================================

    views = convert_to_perspective_views(
        image
    )

    save_perspective_views(
        views
    )

    # =====================================================
    # 8. Scan perspective views
    # =====================================================

    scan_perspective_views(
        views,
        global_results
    )

    # =====================================================
    # 9. Final result
    # =====================================================

    print_results(
        global_results
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()