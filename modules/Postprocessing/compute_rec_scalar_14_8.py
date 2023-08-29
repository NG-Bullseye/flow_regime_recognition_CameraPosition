import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import numpy as np

def main():
    IMAGE_FOLDER = "/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/Bulktraining_Outputs/Lenet5_kroger_5epochs_2023_11_08/yaw/yaw_-19.285715103149414" # Change this to your folder path
    UPSCALE_FACTOR = 10 # Adjust this based on the desired resolution

    image_paths = [os.path.join(IMAGE_FOLDER, filename) for filename in os.listdir(IMAGE_FOLDER) if not os.path.isdir(filename) and "gradcam" in filename]
    print(image_paths)

    image_dict = {image_path: cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for image_path in image_paths}

    fig = plt.figure(figsize=(10, 60))

    for idx, (image_path, img) in enumerate(image_dict.items()):

        upscaled_img = cv2.resize(img, (img.shape[1] * UPSCALE_FACTOR, img.shape[0] * UPSCALE_FACTOR),
                                  interpolation=cv2.INTER_NEAREST)

        grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Print the grayscale image matrix for testing purposes
        print("Grayscale image matrix:")
        print(grayscale_img)

        grayscale_img = cv2.cvtColor(upscaled_img, cv2.COLOR_RGB2GRAY)

        min_val, max_val = np.min(grayscale_img), np.max(grayscale_img)
        normalized_grayscale_img = (grayscale_img - min_val) / (max_val - min_val)

        height, width = normalized_grayscale_img.shape
        total_weight = np.sum(normalized_grayscale_img)
        if total_weight == 0:
            continue

        avg_x = 0
        avg_y = 0
        for y in range(height):
            for x in range(width):
                weight = normalized_grayscale_img[y, x]
                avg_x += (width - x - 1) * weight
                avg_y += (height - y - 1) * weight  # Flipping the y-axis

        avg_x /= total_weight
        avg_y /= total_weight
        center_x, center_y = (width - 1) / 2, (height - 1) / 2

        rect_size = UPSCALE_FACTOR

        # Compute the center of gravity corresponding to the original image by dividing by the upscale factor
        orig_avg_x, orig_avg_y = avg_x / UPSCALE_FACTOR, avg_y / UPSCALE_FACTOR
        print(f"Center of gravity (original coordinates): x = {orig_avg_x:.2f}, y = {orig_avg_y:.2f}")

        # Plotting the normalized grayscale image
        ax1 = fig.add_subplot(len(image_dict), 2, 2 * idx + 1)
        ax1.imshow(normalized_grayscale_img, cmap='gray')
        draw_rectangles_and_lines(ax1, avg_x, avg_y, center_x, center_y, rect_size, title=f"Normalized Center of Gravity Gradient\nx: {center_x:.2f}, y: {center_y:.2f}")

        # Plotting the original upscaled image
        ax2 = fig.add_subplot(len(image_dict), 2, 2*idx+2)
        ax2.imshow(upscaled_img)
        draw_rectangles_and_lines(ax2, avg_x, avg_y, center_x, center_y, rect_size, title=f"OG upscaled Center of Gravity Gradient\nx: {center_x:.2f}, y: {center_y:.2f}")

    plt.tight_layout(pad=0)
    plt.show()

def draw_rectangles_and_lines(ax, avg_x, avg_y, center_x, center_y, rect_size, title):
    rect_gravity = patches.Rectangle((avg_x - rect_size / 2, avg_y - rect_size / 2), rect_size, rect_size, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect_gravity)
    rect_image = patches.Rectangle((center_x - rect_size / 2, center_y - rect_size / 2), rect_size, rect_size, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect_image)
    # Draw a line from the center of the image to the center of gravity
    ax.plot([center_x, avg_x], [center_y, avg_y], color='blue')
    ax.axis('off')
    ax.set_title(title, fontsize=10, loc='center')

if __name__ == '__main__':
    main()
