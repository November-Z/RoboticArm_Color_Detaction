import cv2
import numpy as np

# Camera ID: 0 = default camera, 1 = external camera
CAMERA_ID = 0
cap = cv2.VideoCapture(CAMERA_ID)

# Color HSV ranges
COLORS = {
    "Red": {
        "lower": np.array([0, 120, 80]),
        "upper": np.array([10, 255, 255]),
        "draw": (0, 0, 255)
    },

    "Orange": {
        "lower": np.array([10, 100, 80]),
        "upper": np.array([22, 255, 255]),
        "draw": (0, 165, 255)
    },

    "Yellow": {
        "lower": np.array([22, 100, 80]),
        "upper": np.array([35, 255, 255]),
        "draw": (0, 255, 255)
    },

    "Green": {
        "lower": np.array([35, 80, 60]),
        "upper": np.array([85, 255, 255]),
        "draw": (0, 255, 0)
    },

    "Blue": {
        "lower": np.array([90, 80, 60]),
        "upper": np.array([130, 255, 255]),
        "draw": (255, 0, 0)
    },

    "Balck": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([179, 255, 70]),
        "draw": (80, 80, 80)
    },

    "Dark_Yellow":{
        "lower": np.array([20, 80, 40]),
        "upper": np.array([35, 255, 160]),
        "draw": (0, 180, 180)
    }
}

# Minimum contour area
MIN_AREA = 1000

# Used for noise removal
KERNEL = np.ones((5, 5), np.uint8)

current_hsv = None


# Click image to print HSV value
def mouse_callback(event, x, y, flags, param):
    global current_hsv

    if event == cv2.EVENT_LBUTTONDOWN and current_hsv is not None:
        h, s, v = current_hsv[y, x]

        print("-------------------")
        print("Position:", x, y)
        print("H =", h)
        print("S =", s)
        print("V =", v)
        print("-------------------")


def detect_color(frame, hsv, color_name, lower, upper, draw_color):

    # Create binary mask
    mask = cv2.inRange(hsv, lower, upper)

    # Remove small noise and fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Object center
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            draw_color,
            2
        )

        # Draw center point
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (255, 255, 255),
            -1
        )

        # Show color name
        cv2.putText(
            frame,
            color_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            draw_color,
            2
        )

        # Show center position
        cv2.putText(
            frame,
            f"({center_x},{center_y})",
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            draw_color,
            1
        )

        count += 1

    return count


# Create camera window
cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse_callback)


while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera")
        break

    frame = cv2.resize(frame, (640, 480))

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Used by mouse callback
    current_hsv = hsv

    text_y = 20

    # Detect every color in COLORS
    for color_name, settings in COLORS.items():

        count = detect_color(
            frame,
            hsv,
            color_name,
            settings["lower"],
            settings["upper"],
            settings["draw"]
        )

        cv2.putText(
            frame,
            f"{color_name}: {count}",
            (10, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            settings["draw"],
            2
        )

        text_y += 25

    cv2.putText(
        frame,
        "Click object to read HSV",
        (350, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow("Camera", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
