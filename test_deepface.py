from deepface import DeepFace
result = DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg")
dfs = DeepFace.find(img_path = "img1.jpg", db_path = "C:/workspace/my_db")