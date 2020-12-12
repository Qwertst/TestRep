from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("stars1.csv")
spectra_map = {'M': 0, 'K': 1, 'G': 2, 'F': 3, 'A': 4, 'B': 5, 'O': 6}
StarType = {0: 'Brown Dwarf', 1: 'Red Dwarf', 2: 'White Dwarf', 3: 'Main Sequence', 4: 'Supergiant', 5: 'Hypergiant'}
color_map = {'Red': 0, 'Blue': 1, 'Blue-white': 2, 'White': 3, 'Yellow-white': 4, 'Yellowish': 5, 'Yellowish-white': 6,
             'Orange': 7, 'Whitish': 8, 'White-yellow': 9, 'Pale-yellow-orange': 10, 'Orange-red': 11}

data['Spectral Class'] = data['Spectral Class'].map(spectra_map)
data['Star color'] = data['Star color'].map(color_map)

X = data
X = X.drop(['Star type'], axis=1)
Y = data['Star type']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=666)

clt = DecisionTreeClassifier(max_depth=10)
clt.fit(X_train, Y_train)

fig = plt.figure(figsize=(25, 20))
fig = tree.plot_tree(clt, feature_names=['Temperature (K)', 'Luminosity(L/Lo)',
                                         'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star color', 'Spectral Class'],
                     class_names=['Brown Dwarf', 'Red Dwarf', 'White Dwarf', 'Main Sequence', 'Supergiant',
                                  'Hypergiant '],
                     filled=True)
#plt.savefig("decistion_tree.png")

Inputdata = pd.DataFrame([input().split()], columns=['Temperature (K)', 'Luminosity(L/Lo)',
                                                     'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star color',
                                                     'Spectral Class'])
Y_predict = clt.predict(Inputdata)
print(StarType.get(Y_predict[0]))
