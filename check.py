import cv2
img = cv2.imread(r"C:\Users\praja\Downloads\NnuNetv2-Plant Disease\PlantDisease_nnUNet\nnUNet_raw\Dataset501_PlantSeg\imagesTr\plant_0001_0000.jpg")
mask = cv2.imread(r"C:\Users\praja\Downloads\NnuNetv2-Plant Disease\PlantDisease_nnUNet\nnUNet_raw\Dataset501_PlantSeg\labelsTr\plant_0001.png", cv2.IMREAD_GRAYSCALE)
print("Image:", img.shape, "Mask:", mask.shape)
