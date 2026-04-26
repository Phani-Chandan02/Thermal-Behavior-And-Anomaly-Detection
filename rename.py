import os

folder = "test_images"

for i, filename in enumerate(os.listdir(folder)):
    if filename.endswith(".jpg"):
        new_name = f"img{i+1}.jpg"
        os.rename(
            os.path.join(folder, filename),
            os.path.join(folder, new_name)
        )

print("Renaming done!")