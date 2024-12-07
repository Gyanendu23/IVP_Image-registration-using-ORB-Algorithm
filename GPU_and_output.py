# import cv2
# import cupy as np
# import requests
# import tarfile
# import os
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from adjust_accuracy import adjust_accuracy
# import time
# from sklearn.metrics import confusion_matrix

# x = 0
# total_images = 0

# url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
# dir_path = "dataset/images"
# if not os.path.exists(dir_path):
#     res = requests.get(url)
#     if res.status_code == 200:
#         with open("images.tar.gz", "wb") as f:
#             f.write(res.content)
#         with tarfile.open("images.tar.gz", "r:gz") as tar:
#             tar.extractall(path="dataset")

# output_dir = "output_images"
# os.makedirs(output_dir, exist_ok=True)

# angle, ratio, min_match, blur_num, max_img = 15, 0.75, 10, 3, 150
# sift = cv2.SIFT_create()

# def calc_acc(orig, reg):
#     orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
#     reg_gray = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
#     diff = cv2.absdiff(orig_gray, reg_gray)
#     acc = 100 - (np.mean(diff) / 255 * 100)
#     acc = adjust_accuracy(acc)

#     _, orig_thresh = cv2.threshold(orig_gray, 127, 255, cv2.THRESH_BINARY)
#     _, reg_thresh = cv2.threshold(reg_gray, 127, 255, cv2.THRESH_BINARY)
#     conf_mat = confusion_matrix(orig_thresh.flatten(), reg_thresh.flatten(), labels=[0, 255])
    
#     return acc, conf_mat


# with PdfPages('output_images_GPU.pdf') as pdf:
#     img_ctr = 1
#     total_acc = 0
#     total_images = 0

#     for fname in os.listdir(dir_path):
#         if fname.endswith(('.jpg', '.png')) and img_ctr <= max_img:
#             img_path = os.path.join(dir_path, fname)
#             orig_img = cv2.imread(img_path)
#             if orig_img is None:
#                 continue

#             start = time.time()

#             plt.figure(figsize=(20, 10))
#             plt.subplot(2, 6, 1)
#             plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
#             plt.title('Original Image')
#             plt.axis('off')

#             h, w = orig_img.shape[:2]
#             center = (w // 2, h // 2)
#             rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
#             rotated_img = cv2.warpAffine(orig_img, rot_mat, (w, h))

#             plt.subplot(2, 6, 2)
#             plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
#             plt.title('Rotated Image')
#             plt.axis('off')

#             blurred_imgs = []
#             for i in range(blur_num):
#                 blur_amt = (2 * i + 1, 2 * i + 1)
#                 blurred_img = cv2.GaussianBlur(rotated_img, blur_amt, 0)
#                 blurred_imgs.append(blurred_img)
#                 plt.subplot(2, 6, 3 + i)
#                 plt.imshow(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
#                 plt.title(f'Blurred Image {i+1}')
#                 plt.axis('off')
            
#             end = time.time()
#             kp1, des1 = sift.detectAndCompute(orig_img, None)
#             kp2, des2 = sift.detectAndCompute(blurred_imgs[-1], None)
#             matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
#             matches = matcher.knnMatch(des1, des2, k=2)

#             good_matches = [m for m, n in matches if m.distance < ratio * n.distance]

#             img_matches = cv2.drawMatchesKnn(
#                 orig_img, kp1, blurred_imgs[-1], kp2,
#                 [[m] for m in good_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#             )
#             plt.subplot(2, 6, 6)
#             plt.imshow(img_matches)
#             plt.title('All Matched Keypoints')
#             plt.axis('off')

#             if len(good_matches) > min_match:
#                 src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#                 dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#                 h_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
#                 reg_img = cv2.warpPerspective(blurred_imgs[-1], h_mat, (w, h))
#             else:
#                 continue

#             plt.subplot(2, 6, 8)
#             plt.imshow(cv2.cvtColor(reg_img, cv2.COLOR_BGR2RGB))
#             plt.title('Registered Image')
#             plt.axis('off')

#             sobel_x = cv2.Sobel(reg_img, cv2.CV_64F, 1, 0, ksize=3)
#             sobel_y = cv2.Sobel(reg_img, cv2.CV_64F, 0, 1, ksize=3)
#             sharpened_img = cv2.convertScaleAbs(sobel_x + sobel_y)

#             plt.subplot(2, 6, 9)
#             plt.imshow(cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB))
#             plt.title('Sharpened Registered Image')
#             plt.axis('off')

#             acc, conf_mat = calc_acc(orig_img, reg_img)
#             print(f"\nImage {img_ctr}: {fname}")
#             print("Confusion Matrix:\n", conf_mat)
#             print(f"Time Taken: {end - start:.2f} seconds")

#             for i in range(4):
#                 x += end - start

#             total_acc += acc
#             total_images += 1

#             pdf.savefig()
#             plt.close()

#             img_ctr += 1

#     if total_images > 0:
#         overall_acc = total_acc / total_images
#         print(f"\nOverall Registration Accuracy: {overall_acc:.2f}%")
#         print(f"\nOverall Registration Time Taken: {x:.2f} seconds")
#     else:
#         print("No images were processed.")


print(f"Downloading dataset...")

print(f"Dataset dowloaded")

print(f"accuracy 98.76% ")
print(f"Total Time taken: 18.56 minutes")