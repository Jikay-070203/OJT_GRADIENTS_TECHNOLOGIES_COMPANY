from huggingface_hub import upload_folder

upload_folder(
    folder_path="D:\\Data\\FPT_TOTAL",  # Thư mục chứa model
    path_in_repo="",  # Để trống để upload toàn bộ
    repo_id="hoanguyenthanh15/FPT_TOTAL",  # Thiếu dấu phẩy đã được thêm
    repo_type="dataset",  # Thiếu dấu phẩy đã được thêm
    token=""
)

