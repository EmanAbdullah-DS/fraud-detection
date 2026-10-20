from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from config import SEED
from features import CATEGORICAL, NUMERIC, add_transaction_features


def _features():
    return FunctionTransformer(add_transaction_features)


def logistic_pipeline():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("features", _features()), ("prep", pre), ("model", clf)])


def _tree_prep():
    return ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
    ])


def forest_pipeline():
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                 class_weight="balanced_subsample", random_state=SEED)
    return Pipeline([("features", _features()), ("prep", _tree_prep()), ("model", clf)])


def boosting_pipeline(**params):
    cfg = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
               l2_regularization=0.0, random_state=SEED)
    cfg.update(params)
    return Pipeline([("features", _features()), ("prep", _tree_prep()),
                     ("model", HistGradientBoostingClassifier(**cfg))])
