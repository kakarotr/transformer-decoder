from modelscope.hub.api import HubApi

api = HubApi()
api.login("ms-e2eb8794-9a83-4f6c-9fac-5ee4bba96891")

api.upload_folder(
    repo_id="kakarotter/Calix-Dataset",
    path_in_repo="pretraining/tokenize_corpus",
    folder_path="F:/transformer-decoder/pretraining/tokenize_corpus",
    repo_type="dataset",
)