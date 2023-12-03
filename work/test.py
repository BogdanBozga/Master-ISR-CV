import cv2
import numpy as np
from matplotlib import pyplot as plt
# 2. Download an image and place it in the same directory with the Python project.
img = cv2.imread('city.jpg',3)
# 3. Convert it to grayscale:
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 4. Load the image into a numpy array
img_data = np.asarray(gray_image)
# 5. Perform the 2-D FFT on the image data:
f = np.fft.fft2(img_data)
# 6. Move the zero frequency component to the center of the Fourier spectrum
f = np.fft.fftshift(f)
# 7. Compute the magnitudes (absolute values) of the complex numbers from f
f = abs(f)
# 8. Compute the logarithm for each value to reduce the dynamic range
fourier = np.log10(f)
# 9. Find the minimum values that is a finite number (minimum of an array, ignoring any NaN):
lowest = np.nanmin(fourier[np.isfinite(fourier)])
# 10. Find the maximum values that is a finite number:

highest = np.nanmax(fourier[np.isfinite(fourier)])
# 11. Calculate the original contrast range:
contrast_range = highest - lowest
# 12. Normalize the Fourier spectrum data (“stretch” the contrast)
norm_fourier = (fourier - lowest) / contrast_range * 255
# 13. Display the original image and the fourier image
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1) # (rows, columns, panel number)
ax2 = fig.add_subplot(1, 2, 2) # (rows, columns, panel number)
ax1.imshow(gray_image, cmap = "gray")
ax2.imshow(norm_fourier)
ax1.title.set_text("Original image")
ax1.title.set_text("Fourier image")
plt.show()
# 14. Save the figure (if you wish to)
# fig.savefig('my_figure.png')
# 15. Remove the low frequencies by masking with a rectangular window of size 60x60
rows, cols = norm_fourier.shape
crow, ccol = rows//2 , cols//2
norm_fourier [crow-30:crow+30, ccol-30:ccol+30] = 0
# 16. Apply the inverse shift so that DC component again come at the top-left corner
f_ishift = np.fft.ifftshift(norm_fourier)
# 17. Apply the inverse Fourier Transform:
img_back = np.fft.ifft2(f_ishift)
# 18. The result, again, will be a complex number. You can take its real value.
img_back = np.real(img_back)
# 19. Display the original image and the filtered image
fig2 = plt.figure()
ax1 = fig2.add_subplot(1, 2, 1)
ax2 = fig2.add_subplot(1, 2, 2)
ax1.imshow(gray_image, cmap = "gray")
ax2.imshow(img_back, cmap = "gray")

ax1.title.set_text("Original image")
ax1.title.set_text("Filtered image")
plt.show()