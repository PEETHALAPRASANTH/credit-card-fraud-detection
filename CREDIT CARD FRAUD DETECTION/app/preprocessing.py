from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from .config import FEATURE_COLUMNS, RANDOM_STATE

def build_preprocessor():
    # V1..V28 are already PCA-transformed in the Kaggle dataset.
    # Scale only raw Time and Amount; pass PCA components through unchanged.
    return ColumnTransformer(
        transformers=[("scale_time_amount", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

def build_pipeline(model):
    return ImbPipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", model),
    ])
