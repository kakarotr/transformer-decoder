from modelscope.hub.api import HubApi

api = HubApi()
api.login("ms-b4f6fcf3-3f96-4fd1-946b-8700911a3d1d")

api.upload_folder(
    repo_id="kakarotter/Calix-Dataset",
    path_in_repo="pretraining/tokenize_corpus",
    folder_path="F:/transformer-decoder/pretraining/tokenize_corpus",
    repo_type="dataset",
)