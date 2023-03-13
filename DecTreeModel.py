# Try another model with Decision tree

tree = DecisionTreeClassifier(max_depth=5, random_state=17)
tree.fit(features, survived)

# Validate Model
scores_DecisionTree = cross_val_score(regModel, features, survived, cv=3)
avgAcc_DecisionTree = np.mean(scores)
print(avgAcc_DecisionTree)

# Save Model
mdl = pickle.dumps(regModel)
with open('titanic_DecisionTree.pickle', 'wb') as handle:
	pickle.dump(mdl, handle)