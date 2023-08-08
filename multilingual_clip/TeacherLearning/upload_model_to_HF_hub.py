from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="/home/lenovo/Desktop/arabic_clip/Multilingual-CLIP/multilingual_clip/TeacherLearning/arabic-arabert-Vit-B-32-Transformer-Transformer-PT",
    path_in_repo="arabic-arabert-Vit-B-32-Transformer-Transformer-PT",
    repo_id="pain/bert-base-arabertv2-Vit-B-32-using-tf_3",
    repo_type="model",
)