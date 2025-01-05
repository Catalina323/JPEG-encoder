import matplotlib.pyplot as plt
from scipy import misc
import encoding_decoding as ed
import mse_video as mv

# cerinta 1
X = misc.ascent()
name = "ascent"
ed.gray_encoding(X, name)
ed.gray_decoding(name)


# cerinta 2
X = misc.face()
name = "face"
ed.color_encoding(X, name)
ed.color_decoding(name)


# cerinta 3
X = misc.face()
prag = int(input("introduceti pragul mse: "))
decoded_data, mse_opt, f = mv.find_mse(prag, X, verbose=True)
plt.subplot(121).imshow(X)
plt.subplot(121).set_title("Initial Image")
plt.subplot(122).imshow(decoded_data)
plt.subplot(122).set_title("Decoded Image")
plt.show()


# cerinta 4
video_path = "highway.mp4"
name = mv.video_encoding(video_path, 10)
decoded_frames = mv.video_decoding(name)

