from sklearn.linear_model import LogisticRegression
import numpy as np

x_train, y_train = np.array([[1,2],[2,2],[5,6],[7,8]]), np.array([1,1,0,0])

LR_Model = LogisticRegression()

'''

LogisticRegression(C=1.0, 
                   class_weight=None,
                   dual=False, 
                   fit_intercept=True,
                   intercept_scaling=1, 
                   l1_ratio=None, 
                   max_iter=100,
                   multi_class='auto', 
                   n_jobs=None, 
                   penalty='l2',
                   random_state=None, 
                   solver='lbfgs', 
                   tol=0.0001, 
                   verbose=0,
                   warm_start=False)

'''

LR_Model.fit(x_train, y_train)

y_pred = LR_Model.predict(x_train)
