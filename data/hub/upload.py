from modelscope.hub.api import HubApi

api = HubApi()
api.login("ms-e2eb8794-9a83-4f6c-9fac-5ee4bba96891")

api.upload_folder(
    repo_id="kakarotter/Gllama-Dataset",
    path_in_repo="pretraining/finewiki",
    folder_path="F:/transformer-decoder/pretraining-data/clean/finewiki",
    repo_type="dataset",
)
