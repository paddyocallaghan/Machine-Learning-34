import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import helper_funcs
import feature_builder

subset = {}
subset = helper_funcs.get_subset(helper_funcs.raw_data, subset)
team_season_data = helper_funcs.get_season_table(subset)

## Complete features/Targets
complete_features = feature_builder.get_features(team_season_data, subset, helper_funcs.raw_data)
complete_features_dropna = complete_features.dropna()
complete_targets_dropna = complete_features_dropna['Result']
complete_targets_OH_dropna = complete_features_dropna[(['Win', 'Loss', 'Draw'])]
complete_features_dropna = complete_features_dropna.drop((['Win', 'Loss', 'Draw', 'Result']), axis=1)

X_train, X_test, y_train, y_test = train_test_split(complete_features_dropna, complete_targets_OH_dropna,
                                                    random_state=3)


def random_guess(rows):
    res = np.zeros([rows,3])
    for i in range(rows):
        id = random.randint(0, 2)
        res[i][id] = 1
    return res

y_pred = random_guess(len(y_test))
print("Accuracy score for the random guess model is {}.".format(accuracy_score(y_test,y_pred)))


print("Random_Guess = ",accuracy_score(y_test,y_pred))