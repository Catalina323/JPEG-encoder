import encoding_decoding as ed
import numpy as np
import cv2
import pickle


# sarcina 3
def compute_mse(X_origin, X_jpeg):
    # aducem imaginile la acelasi shape
    # (in unele cazuri shape ul lui X_jpeg poate fi mai mare din cauza padding ului)
    X_origin_ext = np.zeros_like(X_jpeg)
    X_origin_ext = X_origin_ext + X_origin
    return np.sum((X_origin_ext - X_jpeg) ** 2) / (X_jpeg.shape[0] * X_jpeg.shape[1])


def find_mse(prag, X, verbose=False):
    f = 1
    add = 0.07

    while (True):
        encoded_data, huffman_codec, r, c = ed.jpeg_color_encoding(X, f=f)
        decoded_data = ed.jpeg_color_decoding(encoded_data, huffman_codec, r, c)
        mse = compute_mse(X, decoded_data)
        dif = mse - prag

        if verbose:
            print(f"incercam f={f} si obtinem mse:", mse)

        if abs(dif) > 3:
            if dif > 0:
                f -= add
                add /= 2
            else:
                f += add
                add /= 2
        else:
            break

    return decoded_data, mse, f


# sarcina 4
def extract_frames(video_path, n=10):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    frames = []
    huffman_codes = []

    frame_count = 0

    for _ in range(n):  # pentru primele n frameuri
        # while True: # pentru tot videoul dar dureaza aproximativ 12 minute
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        encoded_frame, huffman_codec, r, c = ed.jpeg_color_encoding(frame)
        frames.append(encoded_frame)
        huffman_codes.append(huffman_codec)

        frame_count += 1

    cap.release()

    return frames, huffman_codes, r, c, frame_count


def video_encoding(video_path="highway.mp4", number_of_frames=10):
    print("encoding frames...")

    frames, huffman_codes, r, c, frame_count = extract_frames(video_path, number_of_frames)
    dict_video = {
        "r": r,
        "c": c,
        "frame_count": frame_count,
        "huffman_codes": huffman_codes,
        "data": frames
    }
    name = f"nr_{frame_count}_frames"
    with open(f"{name}.bin", "wb") as file:
        pickle.dump(dict_video, file)

    print("finish encoding.")
    return name


def video_decoding(name):
    print("decoding frames...")
    with open(f"{name}.bin", "rb") as file:
        loaded_data = pickle.load(file)

    r_c = loaded_data["r"]
    c_c = loaded_data["c"]
    frame_count_c = loaded_data["frame_count"]
    huffman_codes_c = loaded_data["huffman_codes"]
    data_c = loaded_data["data"]

    decoded_frames = []
    for i in range(frame_count_c):
        decoded_frame = ed.jpeg_color_decoding(data_c[i], huffman_codes_c[i], r_c, c_c)
        decoded_frames.append(decoded_frame)

    print("finish decoding.")
    return decoded_frames
