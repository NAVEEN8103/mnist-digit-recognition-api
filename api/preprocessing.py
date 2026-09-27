import numpy as np

from PIL import Image


def preprocess_image(image: Image.Image):
    """
    Convert an input image into an MNIST-style 28x28 image.

    Steps:
    1. Convert to grayscale
    2. Detect whether the background is light or dark
    3. Convert to white digit on black background
    4. Crop the digit
    5. Add padding
    6. Resize while preserving aspect ratio
    7. Center the digit
    8. Normalize to 0-1
    9. Return shape (1, 28, 28, 1)
    """

    # --------------------------------------------------------
    # 1. GRAYSCALE
    # --------------------------------------------------------

    image = image.convert("L")

    image_array = np.asarray(
        image,
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # 2. DETECT BACKGROUND
    # --------------------------------------------------------

    h, w = image_array.shape

    corner_size = max(
        1,
        min(h, w) // 10
    )

    corners = np.concatenate([
        image_array[:corner_size, :corner_size].ravel(),
        image_array[:corner_size, -corner_size:].ravel(),
        image_array[-corner_size:, :corner_size].ravel(),
        image_array[-corner_size:, -corner_size:].ravel()
    ])

    background_mean = float(
        np.mean(corners)
    )


    # --------------------------------------------------------
    # 3. WHITE DIGIT ON BLACK BACKGROUND
    # --------------------------------------------------------

    if background_mean > 127:

        # Light background + dark digit
        digit = 255 - image_array

    else:

        # Dark background + light digit
        digit = image_array


    # --------------------------------------------------------
    # 4. REMOVE VERY WEAK BACKGROUND PIXELS
    # --------------------------------------------------------

    digit = np.where(
        digit > 20,
        digit,
        0
    ).astype(np.uint8)


    # --------------------------------------------------------
    # 5. FIND DIGIT BOUNDING BOX
    # --------------------------------------------------------

    ys, xs = np.where(
        digit > 20
    )

    if len(xs) == 0:

        return np.zeros(
            (1, 28, 28, 1),
            dtype=np.float32
        )


    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()


    cropped = digit[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]


    # --------------------------------------------------------
    # 6. PAD THE DIGIT
    # --------------------------------------------------------

    crop_h, crop_w = cropped.shape

    padding = max(
        2,
        int(
            0.15 *
            max(
                crop_h,
                crop_w
            )
        )
    )


    padded = np.pad(
        cropped,
        (
            (padding, padding),
            (padding, padding)
        ),
        mode="constant",
        constant_values=0
    )


    # --------------------------------------------------------
    # 7. RESIZE TO MAXIMUM 20x20
    # --------------------------------------------------------

    pil_digit = Image.fromarray(
        padded
    )


    max_size = 20

    scale = max_size / max(
        pil_digit.width,
        pil_digit.height
    )

    new_width = max(
        1,
        round(
            pil_digit.width * scale
        )
    )

    new_height = max(
        1,
        round(
            pil_digit.height * scale
        )
    )


    resized = pil_digit.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )


    resized_array = np.asarray(
        resized,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # 8. PLACE DIGIT IN 28x28 CANVAS
    # --------------------------------------------------------

    output = np.zeros(
        (28, 28),
        dtype=np.float32
    )


    start_y = (
        28 - new_height
    ) // 2

    start_x = (
        28 - new_width
    ) // 2


    output[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = (
        resized_array / 255.0
    )


    # --------------------------------------------------------
    # 9. CENTER BY CENTER OF MASS
    # --------------------------------------------------------

    total_mass = output.sum()

    if total_mass > 0:

        y_indices, x_indices = np.indices(
            output.shape
        )

        center_y = (
            (y_indices * output).sum()
            / total_mass
        )

        center_x = (
            (x_indices * output).sum()
            / total_mass
        )

        target_center = 13.5

        shift_y = int(
            round(
                target_center - center_y
            )
        )

        shift_x = int(
            round(
                target_center - center_x
            )
        )


        centered = np.zeros_like(
            output
        )


        src_y_start = max(
            0,
            -shift_y
        )

        src_y_end = min(
            28,
            28 - shift_y
        )

        src_x_start = max(
            0,
            -shift_x
        )

        src_x_end = min(
            28,
            28 - shift_x
        )


        dst_y_start = max(
            0,
            shift_y
        )

        dst_y_end = dst_y_start + (
            src_y_end - src_y_start
        )

        dst_x_start = max(
            0,
            shift_x
        )

        dst_x_end = dst_x_start + (
            src_x_end - src_x_start
        )


        centered[
            dst_y_start:dst_y_end,
            dst_x_start:dst_x_end
        ] = output[
            src_y_start:src_y_end,
            src_x_start:src_x_end
        ]


        output = centered


    # --------------------------------------------------------
    # 10. FINAL SHAPE
    # --------------------------------------------------------

    output = output.reshape(
        1,
        28,
        28,
        1
    )


    return output.astype(
        np.float32
    )