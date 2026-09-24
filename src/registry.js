// Catalogue of blocks, settings and scikit-learn models used by the canvas, the
// config panel, the AI agent (as its schema) and the Python engine.
const P = (k, t, d, x = {}) => ({ k, t, d, ...x });
const LOG = { log: true };

// t: float | int | nint (int or None) | cat | bool | str | tuple
export const MODELS = [
  // ---- linear
  { key: 'logreg', name: 'Logistic Regression', fam: 'Linear', tasks: ['classification'], cls: 'sklearn.linear_model:LogisticRegression',
    params: [P('C', 'float', 1, LOG), P('solver', 'cat', 'lbfgs', { o: ['lbfgs', 'liblinear', 'saga', 'newton-cg'] }), P('max_iter', 'int', 2000), P('class_weight', 'cat', null, { o: [null, 'balanced'] })],
    grid: { C: [0.01, 0.1, 1, 10, 100], solver: ['lbfgs', 'liblinear'] } },
  { key: 'ridge_clf', name: 'Ridge Classifier', fam: 'Linear', tasks: ['classification'], cls: 'sklearn.linear_model:RidgeClassifier', params: [P('alpha', 'float', 1, LOG), P('class_weight', 'cat', null, { o: [null, 'balanced'] })], grid: { alpha: [0.01, 0.1, 1, 10, 100] } },
  { key: 'sgd_clf', name: 'SGD Classifier', fam: 'Linear', tasks: ['classification'], cls: 'sklearn.linear_model:SGDClassifier',
    params: [P('loss', 'cat', 'hinge', { o: ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'] }), P('penalty', 'cat', 'l2', { o: ['l2', 'l1', 'elasticnet'] }), P('alpha', 'float', 0.0001, LOG), P('max_iter', 'int', 1000)],
    grid: { loss: ['hinge', 'log_loss', 'modified_huber'], alpha: [1e-5, 1e-4, 1e-3, 1e-2], penalty: ['l2', 'l1', 'elasticnet'] } },
  { key: 'perceptron', name: 'Perceptron', fam: 'Linear', tasks: ['classification'], cls: 'sklearn.linear_model:Perceptron', params: [P('penalty', 'cat', null, { o: [null, 'l2', 'l1', 'elasticnet'] }), P('alpha', 'float', 0.0001, LOG)], grid: { penalty: [null, 'l2', 'l1'], alpha: [1e-5, 1e-4, 1e-3] } },
  { key: 'pa_clf', name: 'Passive Aggressive', fam: 'Linear', tasks: ['classification'], cls: 'sklearn.linear_model:PassiveAggressiveClassifier', params: [P('C', 'float', 1, LOG), P('max_iter', 'int', 1000)], grid: { C: [0.01, 0.1, 1, 10] } },
  { key: 'linreg', name: 'Linear Regression', fam: 'Linear', tasks: ['regression'], cls: 'sklearn.linear_model:LinearRegression', params: [P('fit_intercept', 'bool', true)], grid: { fit_intercept: [true, false] } },
  { key: 'ridge', name: 'Ridge', fam: 'Linear', tasks: ['regression'], cls: 'sklearn.linear_model:Ridge', params: [P('alpha', 'float', 1, LOG)], grid: { alpha: [0.01, 0.1, 1, 10, 100] } },
  { key: 'lasso', name: 'Lasso', fam: 'Linear', tasks: ['regression'], cls: 'sklearn.linear_model:Lasso', params: [P('alpha', 'float', 0.01, LOG), P('max_iter', 'int', 5000)], grid: { alpha: [1e-4, 1e-3, 0.01, 0.1, 1] } },
  { key: 'elasticnet', name: 'Elastic Net', fam: 'Linear', tasks: ['regression'], cls: 'sklearn.linear_model:ElasticNet', params: [P('alpha', 'float', 0.01, LOG), P('l1_ratio', 'float', 0.5, { min: 0, max: 1 }), P('max_iter', 'int', 5000)], grid: { alpha: [1e-3, 0.01, 0.1, 1], l1_ratio: [0.2, 0.5, 0.8] } },
  { key: 'lars', name: 'LARS', fam: 'Linear', tasks: ['regression'], cls: 'sklearn.linear_model:Lars', params: [P('n_nonzero_coefs', 'int', 500)], grid: { n_nonzero_coefs: [5, 10, 50, 500] } },
  { key: 'lassolars', name: 'Lasso LARS', fam: 'Linear', tasks: ['regression'], cls: 'sklearn.linear_model:LassoLars', params: [P('alpha', 'float', 0.01, LOG)], grid: { alpha: [1e-4, 1e-3, 0.01, 0.1] } },
  { key: 'bayes_ridge', name: 'Bayesian Ridge', fam: 'Linear', tasks: ['regression'], cls: 'sklearn.linear_model:BayesianRidge', params: [P('alpha_1', 'float', 1e-6, LOG), P('lambda_1', 'float', 1e-6, LOG)], grid: { alpha_1: [1e-7, 1e-6, 1e-5], lambda_1: [1e-7, 1e-6, 1e-5] } },
  { key: 'ard', name: 'ARD Regression', fam: 'Linear', tasks: ['regression'], cls: 'sklearn.linear_model:ARDRegression', params: [P('alpha_1', 'float', 1e-6, LOG), P('lambda_1', 'float', 1e-6, LOG)], grid: { alpha_1: [1e-7, 1e-6, 1e-5] }, flags: { slow: true } },
  { key: 'huber', name: 'Huber Regressor', fam: 'Linear (robust)', tasks: ['regression'], cls: 'sklearn.linear_model:HuberRegressor', params: [P('epsilon', 'float', 1.35, { min: 1 }), P('alpha', 'float', 0.0001, LOG), P('max_iter', 'int', 1000)], grid: { epsilon: [1.1, 1.35, 1.75, 2.5], alpha: [1e-4, 1e-3, 0.01] } },
  { key: 'ransac', name: 'RANSAC Regressor', fam: 'Linear (robust)', tasks: ['regression'], cls: 'sklearn.linear_model:RANSACRegressor', params: [P('min_samples', 'float', null, { nullable: true }), P('max_trials', 'int', 100)], grid: { max_trials: [50, 100, 200] } },
  { key: 'theilsen', name: 'Theil-Sen Regressor', fam: 'Linear (robust)', tasks: ['regression'], cls: 'sklearn.linear_model:TheilSenRegressor', params: [P('max_subpopulation', 'int', 1000)], grid: { max_subpopulation: [500, 1000, 5000] }, flags: { slow: true } },
  { key: 'sgd_reg', name: 'SGD Regressor', fam: 'Linear', tasks: ['regression'], cls: 'sklearn.linear_model:SGDRegressor', params: [P('loss', 'cat', 'squared_error', { o: ['squared_error', 'huber', 'epsilon_insensitive'] }), P('penalty', 'cat', 'l2', { o: ['l2', 'l1', 'elasticnet'] }), P('alpha', 'float', 0.0001, LOG), P('max_iter', 'int', 2000)], grid: { alpha: [1e-5, 1e-4, 1e-3, 1e-2], loss: ['squared_error', 'huber'] } },
  { key: 'pa_reg', name: 'Passive Aggressive Regressor', fam: 'Linear', tasks: ['regression'], cls: 'sklearn.linear_model:PassiveAggressiveRegressor', params: [P('C', 'float', 1, LOG), P('max_iter', 'int', 1000)], grid: { C: [0.01, 0.1, 1, 10] } },
  { key: 'poisson', name: 'Poisson Regressor', fam: 'Generalized linear', tasks: ['regression'], cls: 'sklearn.linear_model:PoissonRegressor', params: [P('alpha', 'float', 1, LOG), P('max_iter', 'int', 1000)], grid: { alpha: [0.01, 0.1, 1, 10] }, note: 'Needs a non-negative target.' },
  { key: 'tweedie', name: 'Tweedie Regressor', fam: 'Generalized linear', tasks: ['regression'], cls: 'sklearn.linear_model:TweedieRegressor', params: [P('power', 'float', 0, { o: [0, 1, 1.5, 2, 3] }), P('alpha', 'float', 1, LOG), P('max_iter', 'int', 1000)], grid: { power: [0, 1, 1.5, 2], alpha: [0.01, 0.1, 1] } },
  // ---- discriminant / bayes
  { key: 'lda', name: 'Linear Discriminant Analysis', fam: 'Discriminant', tasks: ['classification'], cls: 'sklearn.discriminant_analysis:LinearDiscriminantAnalysis', params: [P('solver', 'cat', 'svd', { o: ['svd', 'lsqr', 'eigen'] })], grid: { solver: ['svd', 'lsqr'] } },
  { key: 'qda', name: 'Quadratic Discriminant Analysis', fam: 'Discriminant', tasks: ['classification'], cls: 'sklearn.discriminant_analysis:QuadraticDiscriminantAnalysis', params: [P('reg_param', 'float', 0.1, { min: 0, max: 1 })], grid: { reg_param: [0.01, 0.1, 0.3, 0.6] } },
  { key: 'gnb', name: 'Gaussian Naive Bayes', fam: 'Naive Bayes', tasks: ['classification'], cls: 'sklearn.naive_bayes:GaussianNB', params: [P('var_smoothing', 'float', 1e-9, LOG)], grid: { var_smoothing: [1e-9, 1e-7, 1e-5, 1e-3] } },
  { key: 'bnb', name: 'Bernoulli Naive Bayes', fam: 'Naive Bayes', tasks: ['classification'], cls: 'sklearn.naive_bayes:BernoulliNB', params: [P('alpha', 'float', 1, LOG)], grid: { alpha: [0.1, 0.5, 1, 2] } },
  { key: 'mnb', name: 'Multinomial Naive Bayes', fam: 'Naive Bayes', tasks: ['classification'], cls: 'sklearn.naive_bayes:MultinomialNB', params: [P('alpha', 'float', 1, LOG)], grid: { alpha: [0.1, 0.5, 1, 2] }, flags: { nonneg: true } },
  { key: 'cnb', name: 'Complement Naive Bayes', fam: 'Naive Bayes', tasks: ['classification'], cls: 'sklearn.naive_bayes:ComplementNB', params: [P('alpha', 'float', 1, LOG)], grid: { alpha: [0.1, 0.5, 1, 2] }, flags: { nonneg: true } },
  // ---- neighbours
  { key: 'knn', name: 'K-Nearest Neighbors', fam: 'Neighbors', tasks: ['classification', 'regression'], cls: { classification: 'sklearn.neighbors:KNeighborsClassifier', regression: 'sklearn.neighbors:KNeighborsRegressor' },
    params: [P('n_neighbors', 'int', 5, { min: 1 }), P('weights', 'cat', 'uniform', { o: ['uniform', 'distance'] }), P('p', 'int', 2, { o: [1, 2] })], grid: { n_neighbors: [3, 5, 11, 21, 31], weights: ['uniform', 'distance'], p: [1, 2] } },
  { key: 'nearest_centroid', name: 'Nearest Centroid', fam: 'Neighbors', tasks: ['classification'], cls: 'sklearn.neighbors:NearestCentroid', params: [P('metric', 'cat', 'euclidean', { o: ['euclidean', 'manhattan'] })], grid: { metric: ['euclidean', 'manhattan'] } },
  // ---- SVM / kernel
  { key: 'svc', name: 'SVM (SVC)', fam: 'SVM', tasks: ['classification'], cls: 'sklearn.svm:SVC',
    params: [P('C', 'float', 1, LOG), P('kernel', 'cat', 'rbf', { o: ['rbf', 'linear', 'poly', 'sigmoid'] }), P('gamma', 'cat', 'scale', { o: ['scale', 'auto'] }), P('degree', 'int', 3), P('probability', 'bool', true), P('class_weight', 'cat', null, { o: [null, 'balanced'] })],
    grid: { C: [0.1, 1, 10, 100], kernel: ['rbf', 'linear'], gamma: ['scale', 0.01, 0.1] }, flags: { slow: true } },
  { key: 'nusvc', name: 'Nu-SVC', fam: 'SVM', tasks: ['classification'], cls: 'sklearn.svm:NuSVC', params: [P('nu', 'float', 0.5, { min: 0.01, max: 1 }), P('kernel', 'cat', 'rbf', { o: ['rbf', 'linear', 'poly', 'sigmoid'] }), P('probability', 'bool', true)], grid: { nu: [0.25, 0.5, 0.75] }, flags: { slow: true } },
  { key: 'linear_svc', name: 'Linear SVC', fam: 'SVM', tasks: ['classification'], cls: 'sklearn.svm:LinearSVC', params: [P('C', 'float', 1, LOG), P('max_iter', 'int', 5000), P('class_weight', 'cat', null, { o: [null, 'balanced'] })], grid: { C: [0.01, 0.1, 1, 10] } },
  { key: 'svr', name: 'SVR', fam: 'SVM', tasks: ['regression'], cls: 'sklearn.svm:SVR', params: [P('C', 'float', 1, LOG), P('kernel', 'cat', 'rbf', { o: ['rbf', 'linear', 'poly', 'sigmoid'] }), P('epsilon', 'float', 0.1), P('gamma', 'cat', 'scale', { o: ['scale', 'auto'] })], grid: { C: [0.1, 1, 10, 100], epsilon: [0.01, 0.1, 0.5], gamma: ['scale', 0.01, 0.1] }, flags: { slow: true } },
  { key: 'nusvr', name: 'Nu-SVR', fam: 'SVM', tasks: ['regression'], cls: 'sklearn.svm:NuSVR', params: [P('nu', 'float', 0.5, { min: 0.01, max: 1 }), P('C', 'float', 1, LOG), P('kernel', 'cat', 'rbf', { o: ['rbf', 'linear', 'poly'] })], grid: { nu: [0.25, 0.5, 0.75], C: [0.1, 1, 10] }, flags: { slow: true } },
  { key: 'linear_svr', name: 'Linear SVR', fam: 'SVM', tasks: ['regression'], cls: 'sklearn.svm:LinearSVR', params: [P('C', 'float', 1, LOG), P('epsilon', 'float', 0.0), P('max_iter', 'int', 5000)], grid: { C: [0.01, 0.1, 1, 10] } },
  { key: 'kernel_ridge', name: 'Kernel Ridge', fam: 'Kernel', tasks: ['regression'], cls: 'sklearn.kernel_ridge:KernelRidge', params: [P('alpha', 'float', 1, LOG), P('kernel', 'cat', 'rbf', { o: ['linear', 'rbf', 'poly', 'laplacian'] })], grid: { alpha: [0.01, 0.1, 1, 10], kernel: ['rbf', 'linear'] }, flags: { slow: true } },
  { key: 'gp_clf', name: 'Gaussian Process Classifier', fam: 'Kernel', tasks: ['classification'], cls: 'sklearn.gaussian_process:GaussianProcessClassifier', params: [P('max_iter_predict', 'int', 100)], grid: { max_iter_predict: [50, 100] }, flags: { slow: true } },
  { key: 'gp_reg', name: 'Gaussian Process Regressor', fam: 'Kernel', tasks: ['regression'], cls: 'sklearn.gaussian_process:GaussianProcessRegressor', params: [P('alpha', 'float', 1e-10, LOG), P('normalize_y', 'bool', true)], grid: { alpha: [1e-10, 1e-5, 1e-2] }, flags: { slow: true } },
  // ---- trees
  { key: 'dt', name: 'Decision Tree', fam: 'Tree', tasks: ['classification', 'regression'], cls: { classification: 'sklearn.tree:DecisionTreeClassifier', regression: 'sklearn.tree:DecisionTreeRegressor' },
    params: [P('max_depth', 'nint', null), P('min_samples_split', 'int', 2, { min: 2 }), P('min_samples_leaf', 'int', 1, { min: 1 }), P('criterion', 'cat', { classification: 'gini', regression: 'squared_error' }, { o: { classification: ['gini', 'entropy', 'log_loss'], regression: ['squared_error', 'friedman_mse', 'absolute_error'] } })],
    grid: { max_depth: [null, 4, 8, 12, 20], min_samples_leaf: [1, 5, 10, 20] } },
  { key: 'extra_tree', name: 'Extra Tree', fam: 'Tree', tasks: ['classification', 'regression'], cls: { classification: 'sklearn.tree:ExtraTreeClassifier', regression: 'sklearn.tree:ExtraTreeRegressor' }, params: [P('max_depth', 'nint', null), P('min_samples_leaf', 'int', 1)], grid: { max_depth: [null, 8, 16], min_samples_leaf: [1, 5, 10] } },
  // ---- ensembles of trees
  { key: 'rf', name: 'Random Forest', fam: 'Ensemble', tasks: ['classification', 'regression'], cls: { classification: 'sklearn.ensemble:RandomForestClassifier', regression: 'sklearn.ensemble:RandomForestRegressor' },
    params: [P('n_estimators', 'int', 200, { min: 1 }), P('max_depth', 'nint', null), P('min_samples_split', 'int', 2, { min: 2 }), P('min_samples_leaf', 'int', 1, { min: 1 }), P('max_features', 'cat', { classification: 'sqrt', regression: 1.0 }, { o: ['sqrt', 'log2', 1.0, 0.5] }), P('bootstrap', 'bool', true), P('class_weight', 'cat', null, { o: [null, 'balanced', 'balanced_subsample'], task: 'classification' })],
    grid: { n_estimators: [100, 200, 400], max_depth: [null, 8, 16], min_samples_leaf: [1, 2, 5], max_features: ['sqrt', 'log2', 0.5] } },
  { key: 'et', name: 'Extra Trees', fam: 'Ensemble', tasks: ['classification', 'regression'], cls: { classification: 'sklearn.ensemble:ExtraTreesClassifier', regression: 'sklearn.ensemble:ExtraTreesRegressor' },
    params: [P('n_estimators', 'int', 200), P('max_depth', 'nint', null), P('min_samples_leaf', 'int', 1), P('max_features', 'cat', { classification: 'sqrt', regression: 1.0 }, { o: ['sqrt', 'log2', 1.0, 0.5] })],
    grid: { n_estimators: [100, 200, 400], max_depth: [null, 8, 16], min_samples_leaf: [1, 2, 5] } },
  { key: 'bagging', name: 'Bagging', fam: 'Ensemble', tasks: ['classification', 'regression'], cls: { classification: 'sklearn.ensemble:BaggingClassifier', regression: 'sklearn.ensemble:BaggingRegressor' }, params: [P('n_estimators', 'int', 20), P('max_samples', 'float', 1.0, { min: 0.1, max: 1 }), P('max_features', 'float', 1.0, { min: 0.1, max: 1 })], grid: { n_estimators: [10, 20, 50], max_samples: [0.5, 0.8, 1.0], max_features: [0.5, 0.8, 1.0] } },
  { key: 'adaboost', name: 'AdaBoost', fam: 'Boosting', tasks: ['classification', 'regression'], cls: { classification: 'sklearn.ensemble:AdaBoostClassifier', regression: 'sklearn.ensemble:AdaBoostRegressor' }, params: [P('n_estimators', 'int', 100), P('learning_rate', 'float', 0.5, LOG)], grid: { n_estimators: [50, 100, 200], learning_rate: [0.05, 0.1, 0.5, 1] } },
  { key: 'gb', name: 'Gradient Boosting', fam: 'Boosting', tasks: ['classification', 'regression'], cls: { classification: 'sklearn.ensemble:GradientBoostingClassifier', regression: 'sklearn.ensemble:GradientBoostingRegressor' },
    params: [P('n_estimators', 'int', 150), P('learning_rate', 'float', 0.1, LOG), P('max_depth', 'int', 3), P('subsample', 'float', 1.0, { min: 0.1, max: 1 }), P('min_samples_leaf', 'int', 1)],
    grid: { n_estimators: [100, 200, 300], learning_rate: [0.03, 0.1, 0.2], max_depth: [2, 3, 5], subsample: [0.8, 1.0] } },
  { key: 'hgb', name: 'Hist Gradient Boosting', fam: 'Boosting', tasks: ['classification', 'regression'], cls: { classification: 'sklearn.ensemble:HistGradientBoostingClassifier', regression: 'sklearn.ensemble:HistGradientBoostingRegressor' },
    params: [P('max_iter', 'int', 200), P('learning_rate', 'float', 0.1, LOG), P('max_leaf_nodes', 'int', 31), P('max_depth', 'nint', null), P('l2_regularization', 'float', 0.0), P('min_samples_leaf', 'int', 20), P('class_weight', 'cat', null, { o: [null, 'balanced'], task: 'classification' })],
    grid: { max_iter: [100, 200, 400], learning_rate: [0.03, 0.1, 0.2], max_leaf_nodes: [15, 31, 63], l2_regularization: [0, 0.1, 1] } },
  { key: 'xgb', name: 'XGBoost', fam: 'Boosting', tasks: ['classification', 'regression'], cls: { classification: 'xgboost:XGBClassifier', regression: 'xgboost:XGBRegressor' },
    params: [P('n_estimators', 'int', 300), P('learning_rate', 'float', 0.1, LOG), P('max_depth', 'int', 6), P('subsample', 'float', 1.0, { min: 0.1, max: 1 }), P('colsample_bytree', 'float', 1.0, { min: 0.1, max: 1 }), P('min_child_weight', 'float', 1), P('reg_lambda', 'float', 1), P('reg_alpha', 'float', 0)],
    grid: { n_estimators: [100, 300, 500], learning_rate: [0.03, 0.1, 0.2], max_depth: [3, 5, 7], subsample: [0.8, 1.0], colsample_bytree: [0.7, 1.0] } },
  { key: 'lgbm', name: 'LightGBM', fam: 'Boosting', tasks: ['classification', 'regression'], cls: { classification: 'lightgbm:LGBMClassifier', regression: 'lightgbm:LGBMRegressor' },
    params: [P('n_estimators', 'int', 300), P('learning_rate', 'float', 0.05, LOG), P('num_leaves', 'int', 31), P('max_depth', 'int', -1), P('subsample', 'float', 1.0), P('colsample_bytree', 'float', 1.0), P('min_child_samples', 'int', 20), P('reg_lambda', 'float', 0)],
    grid: { n_estimators: [100, 300, 500], learning_rate: [0.03, 0.05, 0.1], num_leaves: [15, 31, 63], min_child_samples: [10, 20, 40] } },
  // ---- neural
  { key: 'mlp', name: 'Neural Network (MLP)', fam: 'Neural network', tasks: ['classification', 'regression'], cls: { classification: 'sklearn.neural_network:MLPClassifier', regression: 'sklearn.neural_network:MLPRegressor' },
    params: [P('hidden_layer_sizes', 'tuple', '100'), P('activation', 'cat', 'relu', { o: ['relu', 'tanh', 'logistic'] }), P('alpha', 'float', 0.0001, LOG), P('learning_rate_init', 'float', 0.001, LOG), P('batch_size', 'cat', 'auto', { o: ['auto', 32, 64, 128, 256] }), P('max_iter', 'int', 300), P('early_stopping', 'bool', true)],
    grid: { hidden_layer_sizes: ['64', '128,64', '64,32'], alpha: [1e-4, 1e-3, 1e-2], learning_rate_init: [1e-3, 3e-3] } },
];
export const MODEL = Object.fromEntries(MODELS.map(m => [m.key, m]));
export const FAMILY_COLORS = { Linear: '#2f6bff', 'Linear (robust)': '#3b82f6', 'Generalized linear': '#0284c7', Discriminant: '#0891b2', 'Naive Bayes': '#b45309', Neighbors: '#0e7490', SVM: '#6d3ae0', Kernel: '#7c3aed', Tree: '#65a30d', Ensemble: '#16a34a', Boosting: '#ea8a0c', 'Neural network': '#db2777', Custom: '#64748b', Baseline: '#64748b' };
export const DEFAULT_MODELS = { classification: ['logreg', 'rf', 'xgb', 'svc', 'mlp'], regression: ['ridge', 'rf', 'xgb', 'svr', 'mlp'] };

const CV = ['none', 'kfold', 'stratified_kfold', 'repeated_kfold', 'repeated_stratified_kfold', 'shuffle_split', 'stratified_shuffle_split', 'time_series', 'group_kfold'];
const CLS_METRICS = ['accuracy', 'balanced_accuracy', 'f1', 'f1_weighted', 'precision', 'recall', 'roc_auc', 'average_precision', 'mcc', 'log_loss'];
const REG_METRICS = ['r2', 'rmse', 'mae', 'mape', 'medae', 'explained_variance'];
export { CLS_METRICS, REG_METRICS };

// Settings per block. f: field type (target, col, cols, numcols, cat, int, float, bool, feats, models, json)
// op: shown under "Available operations"; adv: under "Advanced options".
export const BLOCKS = {
  dataset: { t: 'Dataset', d: 'Upload your dataset', c: '#2f6bff', i: 'db', rank: 0, fields: [
    { k: 'target', l: 'Target column', f: 'target', d: '' },
    { k: 'task', l: 'Task', f: 'cat', o: ['auto', 'classification', 'regression'], d: 'auto' },
    { k: 'exclude', l: 'Columns to leave out of training', f: 'cols', d: [] },
    { k: 'group_col', l: 'Group column (for Group K-fold)', f: 'col', d: '', adv: 1 },
    { k: 'time_col', l: 'Time column (sorts rows for time-series splits)', f: 'col', d: '', adv: 1 },
  ] },
  split: { t: 'Validation & CV', d: 'Train / validation / test split, cross-validation', c: '#0369a1', i: 'split', rank: 1, fields: [
    { k: 'test_size', l: 'Test set size', f: 'float', d: 0.2, min: 0.05, max: 0.5, step: 0.05 },
    { k: 'val_size', l: 'Validation set size', f: 'float', d: 0.1, min: 0, max: 0.4, step: 0.05 },
    { k: 'cv_strategy', l: 'Cross-validation', f: 'cat', o: CV, d: 'stratified_kfold' },
    { k: 'folds', l: 'Folds / splits', f: 'int', d: 5, min: 2, max: 20 },
    { k: 'repeats', l: 'Repeats (repeated K-fold)', f: 'int', d: 2, min: 1, max: 10, adv: 1 },
    { k: 'stratify', l: 'Stratify splits by class', f: 'bool', d: true, op: 1 },
    { k: 'shuffle', l: 'Shuffle before splitting', f: 'bool', d: true, op: 1 },
    { k: 'refit_train_val', l: 'Refit final models on train + validation', f: 'bool', d: false, op: 1 },
    { k: 'seed', l: 'Random seed', f: 'int', d: 42, min: 0, max: 99999, adv: 1 },
  ] },
  preprocess: { t: 'Preprocessing', d: 'Clean, handle missing values, encode, scale', c: '#0f9e8a', i: 'gear', rank: 2, fields: [
    { k: 'num_impute', l: 'Missing numbers', f: 'cat', o: ['mean', 'median', 'most_frequent', 'constant', 'knn', 'iterative', 'drop'], d: 'median' },
    { k: 'cat_impute', l: 'Missing categories', f: 'cat', o: ['most_frequent', 'constant'], d: 'most_frequent' },
    { k: 'encoding', l: 'Categorical encoding', f: 'cat', o: ['onehot', 'onehot_drop_first', 'ordinal', 'target', 'frequency'], d: 'onehot' },
    { k: 'scaling', l: 'Scaling', f: 'cat', o: ['standard', 'minmax', 'robust', 'maxabs', 'normalizer', 'quantile_normal', 'quantile_uniform', 'power', 'none'], d: 'standard' },
    { k: 'transform', l: 'Skew transform', f: 'cat', o: ['none', 'log1p', 'yeo-johnson', 'box-cox', 'quantile_normal', 'quantile_uniform'], d: 'none' },
    { k: 'outliers', l: 'Outliers', f: 'cat', o: ['none', 'iqr', 'zscore', 'percentile', 'isolation_forest'], d: 'none' },
    { k: 'imbalance', l: 'Class imbalance', f: 'cat', o: ['none', 'class_weight', 'random_over', 'random_under', 'smote'], d: 'none' },
    { k: 'drop_duplicates', l: 'Remove duplicate rows', f: 'bool', d: true, op: 1 },
    { k: 'drop_constant', l: 'Drop constant columns', f: 'bool', d: true, op: 1 },
    { k: 'drop_high_missing', l: 'Drop columns that are mostly missing', f: 'bool', d: true, op: 1 },
    { k: 'transform_skewed_only', l: 'Only transform skewed columns', f: 'bool', d: true, op: 1 },
    { k: 'fill_value', l: 'Constant fill value', f: 'float', d: 0, adv: 1 },
    { k: 'knn_neighbors', l: 'KNN imputer neighbours', f: 'int', d: 5, min: 1, max: 50, adv: 1 },
    { k: 'max_categories', l: 'Max categories per column (one-hot)', f: 'int', d: 20, min: 2, max: 200, adv: 1 },
    { k: 'high_card_threshold', l: 'High-cardinality threshold', f: 'int', d: 50, min: 5, max: 5000, adv: 1 },
    { k: 'high_card_encoding', l: 'High-cardinality encoding', f: 'cat', o: ['frequency', 'target', 'ordinal', 'drop'], d: 'frequency', adv: 1 },
    { k: 'skew_threshold', l: 'Skewness threshold', f: 'float', d: 1.0, min: 0, max: 10, adv: 1 },
    { k: 'outlier_factor', l: 'Outlier factor (IQR k, z, percentile or contamination)', f: 'float', d: 1.5, min: 0, max: 10, adv: 1 },
    { k: 'drop_missing_threshold', l: 'Drop column if missing share above', f: 'float', d: 0.6, min: 0.1, max: 1, adv: 1 },
  ] },
  fe: { t: 'Feature Engineering', d: 'Create new features, transform data', c: '#7c3aed', i: 'fe', rank: 3, fields: [
    { k: 'custom_features', l: 'Custom features (name = pandas expression)', f: 'feats', d: [] },
    { k: 'polynomial', l: 'Polynomial features', f: 'cat', o: [0, 2, 3], d: 0 },
    { k: 'poly_columns', l: 'Polynomial columns (empty = all numeric)', f: 'numcols', d: [] },
    { k: 'date_parts', l: 'Extract year, month, day, weekday from dates', f: 'bool', d: true, op: 1 },
    { k: 'interaction_only', l: 'Interactions only (no powers)', f: 'bool', d: false, op: 1 },
    { k: 'splines', l: 'Spline features', f: 'bool', d: false, op: 1 },
    { k: 'spline_columns', l: 'Spline columns', f: 'numcols', d: [], adv: 1 },
    { k: 'spline_knots', l: 'Spline knots', f: 'int', d: 5, min: 2, max: 20, adv: 1 },
    { k: 'bin_columns', l: 'Columns to bin (KBinsDiscretizer)', f: 'numcols', d: [], adv: 1 },
    { k: 'n_bins', l: 'Number of bins', f: 'int', d: 5, min: 2, max: 50, adv: 1 },
    { k: 'bin_strategy', l: 'Binning strategy', f: 'cat', o: ['quantile', 'uniform', 'kmeans'], d: 'quantile', adv: 1 },
  ] },
  fs: { t: 'Feature Selection', d: 'Select important features, reduce dimensions', c: '#c026d3', i: 'filter', rank: 4, fields: [
    { k: 'method', l: 'Selection method', f: 'cat', o: ['none', 'kbest_f', 'kbest_mi', 'kbest_chi2', 'percentile_f', 'rfe', 'rfecv', 'l1', 'tree_importance', 'sfs_forward', 'sfs_backward'], d: 'none' },
    { k: 'k', l: 'Features to keep (K)', f: 'int', d: 20, min: 1, max: 1000 },
    { k: 'reduction', l: 'Dimensionality reduction', f: 'cat', o: ['none', 'pca', 'svd', 'ica', 'kernel_pca', 'lda'], d: 'none' },
    { k: 'n_components', l: 'Components (<1 = variance kept for PCA)', f: 'float', d: 0.95, min: 0.01, max: 500 },
    { k: 'variance_filter', l: 'Remove low-variance features', f: 'bool', d: true, op: 1 },
    { k: 'drop_correlated', l: 'Remove highly correlated features', f: 'bool', d: false, op: 1 },
    { k: 'percentile', l: 'Percentile to keep (percentile method)', f: 'int', d: 50, min: 1, max: 100, adv: 1 },
    { k: 'estimator', l: 'Estimator for RFE / RFECV', f: 'cat', o: ['linear', 'rf'], d: 'linear', adv: 1 },
    { k: 'corr_threshold', l: 'Correlation threshold', f: 'float', d: 0.95, min: 0.5, max: 0.999, adv: 1 },
    { k: 'variance_threshold', l: 'Variance threshold', f: 'float', d: 0.0, min: 0, max: 10, adv: 1 },
    { k: 'l1_C', l: 'L1 strength C (classification)', f: 'float', d: 0.5, adv: 1 },
    { k: 'l1_alpha', l: 'L1 alpha (regression)', f: 'float', d: 0.01, adv: 1 },
    { k: 'kernel', l: 'Kernel PCA kernel', f: 'cat', o: ['rbf', 'poly', 'sigmoid', 'cosine'], d: 'rbf', adv: 1 },
  ] },
  model: { t: 'Model', c: '#16a34a', i: 'cpu', rank: 5 },
  zoo: { t: 'Model Zoo', d: 'Train many models at once', c: '#0d9488', i: 'grid', rank: 5, fields: [{ k: 'models', l: 'Models to train', f: 'models', d: [] }] },
  voting: { t: 'Voting Ensemble', d: 'Average the top models', c: '#15803d', i: 'vote', rank: 5.5, fields: [{ k: 'top_k', l: 'Combine the top K models', f: 'int', d: 3, min: 2, max: 10 }, { k: 'voting', l: 'Voting', f: 'cat', o: ['soft', 'hard'], d: 'soft' }] },
  stacking: { t: 'Stacking Ensemble', d: 'Meta-model on top models', c: '#166534', i: 'stack', rank: 5.5, fields: [{ k: 'top_k', l: 'Stack the top K models', f: 'int', d: 3, min: 2, max: 10 }, { k: 'cv', l: 'Internal CV folds', f: 'int', d: 3, min: 2, max: 10 }] },
  eval: { t: 'Model Evaluation', d: 'Metrics, curves, importance, SHAP', c: '#0e7490', i: 'eval', rank: 6, fields: [
    { k: 'primary_metric', l: 'Metric used to rank models', f: 'cat', o: ['auto', ...CLS_METRICS, ...REG_METRICS], d: 'auto' },
    { k: 'permutation', l: 'Permutation importance for', f: 'cat', o: ['none', 'best', 'top3', 'all'], d: 'all' },
    { k: 'shap', l: 'SHAP explanations for', f: 'cat', o: ['none', 'best', 'top3'], d: 'best' },
    { k: 'learning_curve', l: 'Learning curves for', f: 'cat', o: ['none', 'best', 'top3'], d: 'best' },
    { k: 'baseline', l: 'Include a dummy baseline model', f: 'bool', d: true, op: 1 },
    { k: 'perm_repeats', l: 'Permutation repeats', f: 'int', d: 5, min: 1, max: 30, adv: 1 },
    { k: 'shap_rows', l: 'Test rows explained by SHAP', f: 'int', d: 40, min: 5, max: 300, adv: 1 },
  ] },
  tuning: { t: 'Hyperparameter Tuning', d: 'Optuna, Bayesian, grid, random or halving search', c: '#6d28d9', i: 'sliders', rank: 7, fields: [
    { k: 'method', l: 'Search method', f: 'cat', o: ['optuna', 'bayesian', 'random', 'grid', 'halving_grid', 'halving_random'], d: 'optuna' },
    { k: 'models', l: 'Models to tune', f: 'cat', o: ['best', 'top3', 'top5', 'all'], d: 'top3' },
    { k: 'n_iter', l: 'Trials per model (Optuna / Bayesian / random)', f: 'int', d: 20, min: 2, max: 500 },
    { k: 'cv_folds', l: 'CV folds during search', f: 'int', d: 3, min: 2, max: 10 },
    { k: 'enabled', l: 'Tune automatically after training', f: 'bool', d: true, op: 1 },
    { k: 'sampler', l: 'Optuna sampler', f: 'cat', o: ['tpe', 'random', 'qmc'], d: 'tpe', adv: 1 },
    { k: 'timeout', l: 'Time limit per model in seconds (0 = no limit)', f: 'int', d: 0, min: 0, max: 7200, adv: 1 },
    { k: 'max_grid', l: 'Switch grid to random above this many combinations', f: 'int', d: 60, min: 4, max: 5000, adv: 1 },
    { k: 'grids', l: 'Custom search spaces (JSON). Lists or ranges: {"rf": {"n_estimators": {"low": 50, "high": 600, "type": "int"}, "max_depth": [null, 8, 16]}}', f: 'json', d: {}, adv: 1 },
  ] },
  deploy: { t: 'Deployment / Export', d: 'Model file, Python project, API', c: '#be185d', i: 'rocket', rank: 8, fields: [] },
};
export const OPT_LABEL = {
  auto: 'Auto', classification: 'Classification', regression: 'Regression', none: 'None', mean: 'Mean', median: 'Median', most_frequent: 'Most frequent', constant: 'Constant', knn: 'KNN imputer', iterative: 'Iterative imputer (MICE)', drop: 'Drop rows',
  onehot: 'One-hot', onehot_drop_first: 'One-hot (drop first)', ordinal: 'Ordinal', target: 'Target encoding', frequency: 'Frequency',
  standard: 'StandardScaler', minmax: 'MinMaxScaler', robust: 'RobustScaler', maxabs: 'MaxAbsScaler', normalizer: 'Normalizer (row-wise)', quantile_normal: 'Quantile (normal)', quantile_uniform: 'Quantile (uniform)', power: 'PowerTransformer',
  log1p: 'log1p', 'yeo-johnson': 'Yeo-Johnson', 'box-cox': 'Box-Cox', iqr: 'Clip IQR', zscore: 'Clip z-score', percentile: 'Clip percentiles', isolation_forest: 'Isolation Forest (remove rows)',
  class_weight: 'Class weights (balanced)', random_over: 'Random oversampling', random_under: 'Random undersampling', smote: 'SMOTE',
  kfold: 'K-fold', stratified_kfold: 'Stratified K-fold', repeated_kfold: 'Repeated K-fold', repeated_stratified_kfold: 'Repeated stratified K-fold', shuffle_split: 'Shuffle split', stratified_shuffle_split: 'Stratified shuffle split', time_series: 'Time-series split', group_kfold: 'Group K-fold',
  kbest_f: 'SelectKBest (ANOVA F / F-test)', kbest_mi: 'SelectKBest (mutual information)', kbest_chi2: 'SelectKBest (chi²)', percentile_f: 'SelectPercentile (F-test)', rfe: 'RFE', rfecv: 'RFECV', l1: 'L1-based (SelectFromModel)', tree_importance: 'Tree importance (SelectFromModel)', sfs_forward: 'Sequential forward', sfs_backward: 'Sequential backward',
  pca: 'PCA', svd: 'Truncated SVD', ica: 'FastICA', kernel_pca: 'Kernel PCA', lda: 'LDA projection', linear: 'Linear', rf: 'Random forest',
  best: 'Best model', top3: 'Top 3', top5: 'Top 5', all: 'All models', optuna: 'Optuna (Bayesian TPE)', bayesian: 'Bayesian (Gaussian process)', tpe: 'TPE', qmc: 'Quasi-Monte Carlo', random: 'Random search', grid: 'Grid search', halving_grid: 'Halving grid search', halving_random: 'Halving random search',
  accuracy: 'Accuracy', balanced_accuracy: 'Balanced accuracy', f1: 'F1', f1_weighted: 'F1 (weighted)', precision: 'Precision', recall: 'Recall', roc_auc: 'ROC AUC', average_precision: 'Average precision', mcc: 'MCC', log_loss: 'Log loss', kappa: "Cohen's kappa",
  r2: 'R²', rmse: 'RMSE', mae: 'MAE', mape: 'MAPE', medae: 'Median AE', explained_variance: 'Explained variance', max_error: 'Max error', soft: 'Soft', hard: 'Hard', uniform: 'Uniform', kmeans: 'K-means', quantile: 'Quantile',
  0: 'Off', 2: 'Degree 2', 3: 'Degree 3',
};
export const optLabel = o => (o === null ? 'None' : OPT_LABEL[o] ?? String(o));
export const LOWER_BETTER = new Set(['rmse', 'mae', 'mape', 'medae', 'log_loss', 'max_error']);

export function blockDefaults(type) {
  const o = {};
  for (const f of BLOCKS[type]?.fields || []) o[f.k] = Array.isArray(f.d) ? [...f.d] : (f.d && typeof f.d === 'object' ? { ...f.d } : f.d);
  return o;
}
export function modelDefaults(key, task = 'classification') {
  const o = {};
  for (const p of MODEL[key]?.params || []) {
    if (p.o && typeof p.o === 'object' && !Array.isArray(p.o)) continue;
    o[p.k] = p.d && typeof p.d === 'object' ? p.d[task] ?? p.d.classification : p.d;
  }
  return o;
}
export function paramDefault(p, task) { return p.d && typeof p.d === 'object' ? p.d[task] ?? p.d.classification : p.d; }
export function paramOptions(p, task) { return p.o && !Array.isArray(p.o) ? p.o[task] : p.o; }

// Validate a settings patch (from the UI or the AI) against a block schema.
export function sanitizeBlock(type, patch, columns = null) {
  const out = {};
  if (!patch || typeof patch !== 'object') return out;
  for (const f of BLOCKS[type]?.fields || []) {
    if (!(f.k in patch)) continue;
    let v = patch[f.k];
    if (f.f === 'int' || f.f === 'float') { v = Number(v); if (!isFinite(v)) continue; if (f.min != null) v = Math.max(f.min, v); if (f.max != null) v = Math.min(f.max, v); if (f.f === 'int') v = Math.round(v); }
    else if (f.f === 'bool') v = v === true || v === 'true' || v === 1;
    else if (f.f === 'cat') { const hit = f.o.find(o => String(o) === String(v)); if (hit === undefined) continue; v = hit; }
    else if (f.f === 'target' || f.f === 'col') { v = v == null ? '' : String(v); if (v && columns && !columns.includes(v)) continue; }
    else if (f.f === 'cols' || f.f === 'numcols') { if (!Array.isArray(v)) v = [v]; v = v.map(String).filter(c => !columns || columns.includes(c)); }
    else if (f.f === 'feats') { if (!Array.isArray(v)) continue; v = v.filter(x => x && x.name && x.expr).map(x => ({ name: String(x.name).trim().replace(/[^\w]/g, '_').slice(0, 48), expr: String(x.expr).slice(0, 400) })); }
    else if (f.f === 'models') { if (!Array.isArray(v)) continue; v = v.map(String).filter(k => MODEL[k]); }
    else if (f.f === 'json') { if (typeof v === 'string') { try { v = JSON.parse(v); } catch { continue; } } if (typeof v !== 'object') continue; }
    out[f.k] = v;
  }
  return out;
}
export function sanitizeModelParams(key, patch, task) {
  const out = {};
  for (const p of MODEL[key]?.params || []) {
    if (!(p.k in (patch || {}))) continue;
    let v = patch[p.k];
    if (v === null || v === 'None' || v === '') { if (p.t === 'nint' || p.nullable || (paramOptions(p, task) || []).includes(null)) out[p.k] = null; continue; }
    if (p.t === 'int' || p.t === 'nint') { v = Math.round(Number(v)); if (!isFinite(v)) continue; }
    else if (p.t === 'float') { v = Number(v); if (!isFinite(v)) { const opts = paramOptions(p, task); if (!opts || !opts.includes(patch[p.k])) continue; v = patch[p.k]; } }
    else if (p.t === 'bool') v = v === true || v === 'true';
    else if (p.t === 'tuple') { v = String(v).replace(/[()\[\]\s]/g, ''); if (!/^\d+(,\d+)*$/.test(v)) continue; }
    else if (p.t === 'cat') { const opts = paramOptions(p, task) || []; const hit = opts.find(o => String(o) === String(v)); if (hit === undefined && !isFinite(Number(v))) continue; v = hit !== undefined ? hit : Number(v); }
    if (p.min != null && typeof v === 'number') v = Math.max(p.min, v);
    if (p.max != null && typeof v === 'number') v = Math.min(p.max, v);
    out[p.k] = v;
  }
  return out;
}
// Tuple parameters travel as arrays so Python can turn them into tuples.
export function paramsForEngine(key, params) {
  const out = { ...params };
  for (const p of MODEL[key]?.params || []) if (p.t === 'tuple' && typeof out[p.k] === 'string') out[p.k] = out[p.k].split(',').map(Number).filter(n => n > 0);
  return out;
}
export function gridForEngine(key, grid) {
  const out = {};
  for (const [k, vals] of Object.entries(grid || {})) {
    const p = MODEL[key]?.params.find(x => x.k === k);
    out[k] = p && p.t === 'tuple' && Array.isArray(vals) ? vals.map(v => (typeof v === 'string' ? v.split(',').map(Number) : v)) : vals;
  }
  return out;
}
export function modelSpec(id, key, params, task) {
  const m = MODEL[key];
  const space = Object.fromEntries(m.params.map(p => [p.k, { type: p.t, log: !!p.log, cat: !!p.o || p.t === 'tuple' || p.t === 'bool' }]));
  return { id, key, name: m.name, family: m.fam, cls: typeof m.cls === 'string' ? m.cls : m.cls, tasks: m.tasks, params: paramsForEngine(key, params), grid: gridForEngine(key, m.grid), space, flags: m.flags || {} };
}
