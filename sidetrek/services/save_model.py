import joblib
import bentoml

with open("/userRepoPath/example_owner/example_repo/sidetrek/models/example_model_file", "rb") as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model("example_model", model)
    print(saved_model) # This is required!
