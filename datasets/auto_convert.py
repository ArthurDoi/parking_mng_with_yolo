import os
from ultralytics import SAM

from all_centers import image_centers


MODEL_PATH = "sam2.1_b.pt"
IMAGE_DIR = "images_frame"


model = SAM(MODEL_PATH)

model.info()


for image_name, value in image_centers.items():

    input_image = os.path.join(
        IMAGE_DIR,
        image_name
    )

    if not os.path.exists(input_image):
        print(f"Không tìm thấy ảnh: {input_image}")
        continue
    points = []

    for arr in value:
        for point in arr:
            points.append([
                int(point[0]),
                int(point[1])
            ])


    print("=" * 50)
    print(f"Image      : {input_image}")
    print(f"Num points : {len(points)}")

    results = model(
        input_image,
        points=points
    )
    output_txt = os.path.splitext(input_image)[0] + ".txt"


    with open(output_txt, "w", encoding="utf-8") as f:

        for result in results:

            normalized_bboxes = result.boxes.xywhn

            for bbox in normalized_bboxes:

                x, y, w, h = bbox.tolist()

                f.write(
                    f"0 {x} {y} {w} {h}\n"
                )


    print(f"Output     : {output_txt}")


print("=" * 50)
print("Hoàn thành!")
