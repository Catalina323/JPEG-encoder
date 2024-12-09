import matplotlib.pyplot as plt
from scipy import misc, ndimage
import mse_video as mv


# sarcina 3
X = misc.face()
prag = int(input("alegeti pragul mse: "))

decoded_data, mse_opt, f = mv.find_mse(prag, X, verbose=True)

plt.subplot(121).imshow(X)
plt.subplot(121).set_title("original image")
plt.subplot(122).imshow(decoded_data)
plt.subplot(122).set_title("decoded image")
plt.show()


# sarcina 4
video_path = "highway.mp4"
output_folder = "frames"
number_of_frames = 10
encoded_frames, r, c = mv.extract_encoded_frames(video_path, output_folder, number_of_frames)
decoded_frames = mv.decode_frames(encoded_frames, r, c)

count = 1
for decoded_frame in decoded_frames:
    plt.imshow(decoded_frame)
    plt.title(f"Frame {count}")
    plt.show()
    count += 1