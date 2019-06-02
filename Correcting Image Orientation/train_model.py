from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import argparse
import pickle
import h5py

# construct the argument parse and parse the arguments
# --db is the path to our input HDF5 dataset
# --model is the path to our output serialized Logistic Regression classifier once it has been trained.
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,
            help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,
            help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,
            help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

# construct a training (75%) and testing (25%) dataset based on the number of records in the db
db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75) # index to split the dataset

# using GridSearchCv to tune params for Logistic Regression classifier
# print("Tuning hyperparameters...")
# params = {"C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}

# model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args["jobs"])
# model.fit(db["features"][:i], db["labels"][:i])
# print("Best hyperparameters: {}".format(model.best_estimator_))

model = LogisticRegression(C=0.01).fit(db["features"][:i], db["labels"][:i])

# evaluate the model
print("Evaluating...")
preds = model.predict(db["features"][i:])

print(classification_report(db["labels"][i:], preds,
                        target_names=db["label_names"]))


# serialize the model to disk
print("Saving model...")
f = open(args["model"], "wb")
# f.write(pickle.dumps(model.best_estimator_))
f.write(pickle.dumps(model))
f.close()

# close the db
db.close()

