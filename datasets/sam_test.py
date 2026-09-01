import cv2
from ultralytics import SAM

# Load a model
# For SAM=sam_b.pt, SAM 2=sam2_b.pt, SAM 2.1=sam2.1_b.pt (default, runs out-of-the-box)
# Upgrade to SAM 3 by setting model="sam3.pt". SAM 3 weights require approval + download from
# https://huggingface.co/facebook/sam3 — place sam3.pt in your working directory first.
model = SAM("sam2.1_b.pt")

model.info()  # Display model information (optional)
input_image = "cars\\images\\train\\stage1_t0s.jpg"
# Run inference (image or video)
results = model(input_image, points=[[1352, 194], [1431, 197], [1497, 191], [1579, 181], [1645, 181], [1715, 197]])# image
# results = model("https://youtu.be/LNwODJXcvt4")  # video file

# for i, res in enumerate(results):
#     normalized_bboxes = res.boxes.xywhn
#     with open(input_image.replace(".jpg", ".txt"),"w", encoding="utf-8") as f:
#         for nbbox in normalized_bboxes:
#             x,y,w,h = nbbox
#             f.write("0 {} {} {} {}".format(x,y,w,h)+"\n")

image = results[0].plot(labels=False)
#
cv2.namedWindow("test", cv2.WINDOW_NORMAL)   # cho phép resize cửa sổ
cv2.resizeWindow("test", 1280, 720)          # đặt kích thước cửa sổ vừa màn hình
cv2.imshow("test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()