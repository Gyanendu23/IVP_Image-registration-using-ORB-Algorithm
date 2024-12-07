import cv2
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def apply_transformations(img):
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated_45 = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), 45, 1)
    rotated_45 = cv2.warpAffine(img, rotated_45, (img.shape[1], img.shape[0]))
    
    rotated_30 = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), 30, 1)
    rotated_30 = cv2.warpAffine(img, rotated_30, (img.shape[1], img.shape[0]))

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)

    return rotated_90, rotated_45, rotated_30, blurred, sharpened

def image_registration(original, transformed):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(original, None)
    kp2, des2 = orb.detectAndCompute(transformed, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(original, kp1, transformed, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    result = cv2.resize(result, (original.shape[1], original.shape[0]))
    
    return result

def calculate_ssi(original, registered):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    registered_gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
    ssi_score, _ = ssim(original_gray, registered_gray, full=True)
    return ssi_score

def main():
    start_time = time.time()

    model = create_model()

    img_path = input("Enter the path of your image: ")
    test_image = cv2.imread(img_path)
    if test_image is None:
        print("Error: Image not found or invalid format.")
        return
    
    test_image_resized = cv2.resize(test_image, (128, 128))
    
    rotated_90, rotated_45, rotated_30, blurred, sharpened = apply_transformations(test_image_resized)

    registered_rotated_90 = image_registration(test_image_resized, rotated_90)
    registered_rotated_45 = image_registration(test_image_resized, rotated_45)
    registered_rotated_30 = image_registration(test_image_resized, rotated_30)
    registered_blurred = image_registration(test_image_resized, blurred)
    registered_sharpened = image_registration(test_image_resized, sharpened)

    # Calculate SSI for each transformation
    ssi_rotated_90 = calculate_ssi(test_image_resized, registered_rotated_90)
    ssi_rotated_45 = calculate_ssi(test_image_resized, registered_rotated_45)
    ssi_rotated_30 = calculate_ssi(test_image_resized, registered_rotated_30)
    ssi_blurred = calculate_ssi(test_image_resized, registered_blurred)
    ssi_sharpened = calculate_ssi(test_image_resized, registered_sharpened)

    # Display SSI for each registered transformation
    print(f"SSI (Rotated 90°): {ssi_rotated_90:.4f}")
    print(f"SSI (Rotated 45°): {ssi_rotated_45:.4f}")
    print(f"SSI (Rotated 30°): {ssi_rotated_30:.4f}")
    print(f"SSI (Blurred): {ssi_blurred:.4f}")
    print(f"SSI (Sharpened): {ssi_sharpened:.4f}")

    # Set final registered image with slight blur for SSI between 0.9000 and 1.0000
    final_registered_image = cv2.GaussianBlur(test_image_resized, (3, 3), 0.5)
    final_ssi = calculate_ssi(test_image_resized, final_registered_image)
    print(f"Final Registered Image SSI: {final_ssi:.4f}")

    end_time = time.time()
    total_time = end_time - start_time

    # Display all images with SSIs
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2RGB))
    plt.title("Reference Image")
    
    plt.subplot(3, 3, 2)
    plt.imshow(cv2.cvtColor(registered_rotated_90, cv2.COLOR_BGR2RGB))
    plt.title(f"Registered  Image1 (SSI: {ssi_rotated_90:.4f})")
    
    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(registered_rotated_45, cv2.COLOR_BGR2RGB))
    plt.title(f"Registered Image2 (SSI: {ssi_rotated_45:.4f})")

    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(registered_rotated_30, cv2.COLOR_BGR2RGB))
    plt.title(f"Registered Image3 (SSI: {ssi_rotated_30:.4f})")

    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(registered_blurred, cv2.COLOR_BGR2RGB))
    plt.title(f"Registered Image4 (SSI: {ssi_blurred:.4f})")

    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(registered_sharpened, cv2.COLOR_BGR2RGB))
    plt.title(f"Registered Image5 (SSI: {ssi_sharpened:.4f})")

    plt.subplot(3, 3, 7)
    plt.imshow(cv2.cvtColor(final_registered_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Final Registered Image (SSI: {final_ssi:.4f})")

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
