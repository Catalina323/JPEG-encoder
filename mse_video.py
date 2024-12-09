import encoding_decoding as ed
import numpy as np
import cv2
import os

# sarcina 3
def compute_mse(X_origin, X_jpeg):
    # aducem imaginile la acelasi shape
    # (in unele cazuri shape ul lui X_jpeg poate fi mai mare din cauza padding ului)
    X_origin_ext = np.zeros_like(X_jpeg)
    X_origin_ext = X_origin_ext + X_origin
    return np.sum((X_origin_ext - X_jpeg) ** 2) / (X_jpeg.shape[0] * X_jpeg.shape[1])


def find_mse(prag, X, verbose=False):
    ff = 1
    add = 0.07

    while (True):
        encoded_data, r, c = ed.jpeg_color_encoding(X, f=ff)
        decoded_data = ed.jpeg_color_decoding(encoded_data, r, c)
        mse = compute_mse(X, decoded_data)
        dif = mse - prag

        if verbose:
            print(f"incercam factor={ff} si obtinem mse: ", mse)

        if abs(dif) > 3:
            if dif > 0:
                ff = ff - add
                add /= 2
            else:
                ff = ff + add
                add /= 2
        else:
            break

    return decoded_data, mse, ff


# sarcina 4
def extract_encoded_frames(video_path, output_folder, n=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    frames = []

    frame_count = 0
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        encoded_frame, r, c = ed.jpeg_color_encoding(frame)
        frames.append(encoded_frame)
        frame_count += 1

    cap.release()

    return frames, r, c

def decode_frames(encoded_frames, r, c):
    decoded_frames = []
    for frame in encoded_frames:
        decoded_frame = ed.jpeg_color_decoding(frame, r, c)
        decoded_frames.append(decoded_frame)
    return decoded_frames



