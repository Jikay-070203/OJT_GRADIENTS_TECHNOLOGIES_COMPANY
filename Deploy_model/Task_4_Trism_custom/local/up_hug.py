from huggingface_hub import upload_folder

upload_folder(
    folder_path="D:\\Data\\FPT_TOTAL", 
    repo_id="hoanguyenthanh15/FPT_TOTAL", 
    repo_type="dataset",  
    token=""
)

