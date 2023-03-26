import joblib
import bentoml

with open("/userRepoData/taeefnajib/glass-type-test/sidetrek/models/e9b0286a454049ac88002c2adc030a48.joblib", "rb") as f:
    model = joblib.load(f)
    saved_model = bentoml.sklearn.save_model("example_model", model)
    print(saved_model) # This is required!

