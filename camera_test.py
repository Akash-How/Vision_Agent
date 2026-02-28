import cv2


def test_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Testing camera {index}: Failed")
        cap.release()
        return

    print(f"Testing camera {index}: Success")
    window_name = f"Camera {index} - Press ESC to close"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyWindow(window_name)


def main():
    for idx in range(6):
        test_camera(idx)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
