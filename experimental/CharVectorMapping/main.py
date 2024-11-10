import os
# ctrl images 
image_contorols = "experimental/CharVectorMapping/images/dataset/ctrls"
image_positives = "experimental/CharVectorMapping/images/dataset/positives"
image_ctrls_paths = [
    os.path.join(image_contorols, file) for file in os.listdir(image_contorols) if file.endswith(".png")
]
image_positives_paths = [
    os.path.join(image_positives, file) for file in os.listdir(image_positives) if file.endswith(".png")
]

