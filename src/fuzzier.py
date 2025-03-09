import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import json

from sklearn.metrics import mean_squared_error

def trapezoidal_mf(x, a, b, c, d):
    eps = 1e-6
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + eps), 1), (d - x) / (d - c + eps)))

def gaussian_mf(x, mean, sigma):
    result = np.exp(-0.5 * ((x - mean) / (sigma+1e-6)) ** 2)
    return np.where(result < 0.01, 0, result)

def edge_gaussian_mf(x, mean, sigma, is_right_edge=True):
    if is_right_edge:
        return np.where(x >= mean, 1.0, gaussian_mf(x, mean, sigma))
    else:
        return np.where(x <= mean, 1.0, gaussian_mf(x, mean, sigma))
    
def find_gaussian_intersection(centers, stdevs):
    cross_points = []
    for i in range(len(centers)-1):
        c1, c2 = centers[i], centers[i+1]
        s1, s2 = stdevs[i], stdevs[i+1]
        
        # Quadratic equation coefficients
        a = 1/(2*s1**2 + 1e-6) - 1/(2*s2**2 + 1e-6)
        b = -c1/s1**2 + c2/(s2**2 + 1e-6)
        c = c1**2/(2*s1**2 + 1e-6) - c2**2/(2*s2**2 + 1e-6) - np.log(s2/(s1+1e-6))
        
        # Solve quadratic equation
        if a != 0:
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                x1 = (-b + np.sqrt(discriminant))/(2*a)
                x2 = (-b - np.sqrt(discriminant))/(2*a)
                # Take point between centers
                cross_point = x1 if c1 < x1 < c2 else x2
                if c1 < cross_point < c2:
                    cross_points.append(cross_point)
        
                else:
                    cross_points.append((c1 * s1 + c2 * s2)/(s1 + s2))
        else:
            cross_points.append((c1 * s1 + c2 * s2)/(s1 + s2))
    return cross_points
    
    
class FuzzyRegression:
    def __init__(self, 
                 input_columns, 
                 output_column, 
                 std_factor=1/3, 
                 fuzzy_type='type1', 
                 fuzzy_shape="All gaussian", 
                 number_of_fuzzy_sets="optimize", 
                 kmeans_dim = 1,
                 kmeans_max=5):
        self.input_columns = input_columns
        self.output_column = output_column
        self.fuzzy_type = fuzzy_type
        self.fuzzy_vars = None
        self.rules = None
        self.cross_rules = {}

        self.fuzzy_shape = fuzzy_shape
        self.std_factor = std_factor
        self.number_of_fuzzy_sets = number_of_fuzzy_sets
        self.kmeans_max = kmeans_max
        self.kmeans_dim = kmeans_dim
        self._load_fuzzy_names()

    def _load_fuzzy_names(self, file_path="src/fuzzy_set_names.txt"):
        with open(file_path, "r") as f:
            self.fuzzy_names = json.load(f)

    def _generate_fuzzy_sets_type1(self, centers, stdev, names):
        fuzzy_sets = {}
        for i, name in enumerate(names):
            if i == 0:
                fuzzy_sets[name] = lambda x, c=centers[0], s=stdev[0]: edge_gaussian_mf(x, c, s, is_right_edge=False)
            elif i == len(names)-1:
                fuzzy_sets[name] = lambda x, c=centers[-1], s=stdev[-1]: edge_gaussian_mf(x, c, s, is_right_edge=True)
            else:
                fuzzy_sets[name] = lambda x, c=centers[i], s=stdev[i]: gaussian_mf(x, c, s)
        return fuzzy_sets

    def _generate_fuzzy_sets_type2(self, centers, stdev, names, std_factor):
        fuzzy_sets = {}
        for i, name in enumerate(names):
            if i == 0:
                fuzzy_sets[f"{name}_upper"] = lambda x, c=centers[0], s=stdev[0]: edge_gaussian_mf(x, c, s, is_right_edge=False)
                fuzzy_sets[f"{name}_lower"] = lambda x, c=centers[0], s=stdev[0]: edge_gaussian_mf(x, c, s*std_factor, is_right_edge=False)
            elif i == len(names)-1:
                fuzzy_sets[f"{name}_upper"] = lambda x, c=centers[-1], s=stdev[-1]: edge_gaussian_mf(x, c, s, is_right_edge=True)
                fuzzy_sets[f"{name}_lower"] = lambda x, c=centers[-1], s=stdev[-1]: edge_gaussian_mf(x, c, s*std_factor, is_right_edge=True)
            else:
                fuzzy_sets[f"{name}_upper"] = lambda x, c=centers[i], s=stdev[i]: gaussian_mf(x, c, s)
                fuzzy_sets[f"{name}_lower"] = lambda x, c=centers[i], s=stdev[i]: gaussian_mf(x, c, s*std_factor)
        return fuzzy_sets

    def create_fuzzy_sets(self, X_train, y_train):
        fuzzy_vars = {}
        for col in self.input_columns:
            col_data = X_train[col].values
            col_data_y = y_train.values
            if self.number_of_fuzzy_sets == 'optimize':
                best_inertia = float('inf') 

                if self.kmeans_dim == 1:
                    km = KMeans(n_clusters=self.kmeans_max, n_init=10, random_state=42, algorithm='lloyd').fit(col_data.reshape(-1, 1))
                else:
                    km = KMeans(n_clusters=self.kmeans_max, n_init=10, random_state=42, algorithm='lloyd').fit(np.column_stack((col_data, col_data_y)))
                
                centers = sorted(km.cluster_centers_[:,0].flatten())
                stdev = [(centers[i+1] - centers[i-1]) / 2 * self.std_factor*4 for i in range(1, len(centers) - 1)]
                stdev = [np.std(col_data) * self.std_factor] + stdev + [np.std(col_data) * self.std_factor]
                cross_points = find_gaussian_intersection(centers, stdev)
                self.cross_rules[col] = cross_points
                names = self.fuzzy_names[f'{len(centers)}']
                if self.fuzzy_type == 'type1':
                    fuzzy_vars[col] = self._generate_fuzzy_sets_type1(centers, stdev, names)
                else:
                    fuzzy_vars[col] = self._generate_fuzzy_sets_type2(centers, stdev, names, self.std_factor)
            elif self.number_of_fuzzy_sets == 'FCM':
                mean = np.mean(col_data)
                std = np.std(col_data) * self.std_factor
                q1, q2, q3 = np.percentile(col_data, [25, 50, 75])
                cross_points = find_gaussian_intersection([q1, q2, q3], [std, std, std])
                self.cross_rules[col] = cross_points
                if self.fuzzy_type == 'type1':
                    if self.fuzzy_shape == "All gaussian":
                        fuzzy_vars[col] = {
                            'low': lambda x: edge_gaussian_mf(x, q1, std, False),
                            'medium': lambda x: gaussian_mf(x, q2, std),
                            'high': lambda x: edge_gaussian_mf(x, q3, std, True)
                        }
                    else:
                        fuzzy_vars[col] = {
                            'low': lambda x: trapezoidal_mf(x, 0, 0, q1, q1 + std),
                            'medium': lambda x: gaussian_mf(x, q2, std),
                            'high': lambda x: trapezoidal_mf(x, q3 - std, q3, 1, 1)
                        }
                else:
                    std_mod = std * self.std_factor
                    if self.fuzzy_shape == "All gaussian":
                        fuzzy_vars[col] = {
                            'low_upper': lambda x: edge_gaussian_mf(x, q1, std, False),
                            'low_lower': lambda x: edge_gaussian_mf(x, q1, std_mod, False),
                            'medium_upper': lambda x: gaussian_mf(x, q2, std),
                            'medium_lower': lambda x: gaussian_mf(x, q2, std_mod),
                            'high_upper': lambda x: edge_gaussian_mf(x, q3, std, True),
                            'high_lower': lambda x: edge_gaussian_mf(x, q3, std_mod, True)
                        }
                    else:
                        fuzzy_vars[col] = {
                            'low_upper': lambda x: trapezoidal_mf(x, 0, 0, q1, q1 + std),
                            'low_lower': lambda x: trapezoidal_mf(x, 0, 0, q1, q1 + std_mod),
                            'medium_upper': lambda x: gaussian_mf(x, q2, std),
                            'medium_lower': lambda x: gaussian_mf(x, q2, std_mod),
                            'high_upper': lambda x: trapezoidal_mf(x, q3 - std, q3, 1, 1),
                            'high_lower': lambda x: trapezoidal_mf(x, q3 - std_mod, q3, 1, 1)
                        }

            else:
                mean = np.mean(col_data)
                std = np.std(col_data)*self.std_factor
                q1, q2, q3 = np.percentile(col_data, [25, 50, 75])
                cross_points = find_gaussian_intersection([q1, q2, q3], [std,std,std])
                self.cross_rules[col] = cross_points
                if self.fuzzy_type == 'type1':
                    if self.fuzzy_shape == "All gaussian":
                        fuzzy_vars[col] = {
                            'low': lambda x: edge_gaussian_mf(x, q1, std, False),
                            'medium': lambda x: gaussian_mf(x, q2, std),
                            'high': lambda x: edge_gaussian_mf(x, q3, std, True)
                        }
                    else:
                        fuzzy_vars[col] = {
                            'low': lambda x: trapezoidal_mf(x, 0, 0, q1, q1 + std),
                            'medium': lambda x: gaussian_mf(x, q2, std),
                            'high': lambda x: trapezoidal_mf(x, q3 - std, q3, 1, 1)
                        }
                else:
                    std_mod = std*self.std_factor
                    if self.fuzzy_shape == "All gaussian":
                        fuzzy_vars[col] = {
                            'low_upper': lambda x: edge_gaussian_mf(x, q1, std, False),
                            'low_lower': lambda x: edge_gaussian_mf(x, q1, std_mod, False),
                            'medium_upper': lambda x: gaussian_mf(x, q2, std),
                            'medium_lower': lambda x: gaussian_mf(x, q2, std_mod),
                            'high_upper': lambda x: edge_gaussian_mf(x, q3, std, True),
                            'high_lower': lambda x: edge_gaussian_mf(x, q3, std_mod, True)
                        }
                    else:
                        fuzzy_vars[col] = {
                            'low_upper': lambda x: trapezoidal_mf(x, 0, 0, q1, q1 + std),
                            'low_lower': lambda x: trapezoidal_mf(x, 0, 0, q1, q1 + std_mod),
                            'medium_upper': lambda x: gaussian_mf(x, q2, std),
                            'medium_lower': lambda x: gaussian_mf(x, q2, std_mod),
                            'high_upper': lambda x: trapezoidal_mf(x, q3 - std, q3, 1, 1),
                            'high_lower': lambda x: trapezoidal_mf(x, q3 - std_mod, q3, 1, 1)
                        }
        self.fuzzy_vars = fuzzy_vars

    def create_rules(self, X_train, y_train):
        rules_dict = {}
        for index, row in X_train.iterrows():
            rule = {}
            if self.fuzzy_type == 'type2':
                for col in self.input_columns:
                    keys = [k for k in self.fuzzy_vars[col].keys() if '_upper' in k or '_lower' in k]
                    cat_vals = {}
                    cat_groups = set([k.rsplit('_', 1)[0] for k in keys])
                    for grp in cat_groups:
                        up = self.fuzzy_vars[col][f'{grp}_upper'](row[col])
                        lo = self.fuzzy_vars[col][f'{grp}_lower'](row[col])
                        cat_vals[grp] = (up + lo) / 2
                    rule[col] = max(cat_vals, key=cat_vals.get)
                rule_key = tuple(rule[col] for col in self.input_columns)
                if rule_key in rules_dict:
                    curr_min, curr_max = rules_dict[rule_key]['output']
                    rules_dict[rule_key] = {
                        **rule,
                        'output': (
                            min(curr_min, y_train.loc[index]),
                            max(curr_max, y_train.loc[index])
                        )
                    }
                else:
                    output_val = y_train.loc[index]
                    rules_dict[rule_key] = {**rule, 'output': (output_val, output_val)}
            else:
                for col in self.input_columns:
                    membership = {}
                    for k in self.fuzzy_vars[col].keys():
                        membership[k] = self.fuzzy_vars[col][k](row[col])
                    rule[col] = max(membership, key=membership.get)
                rule_key = tuple(rule[col] for col in self.input_columns)
                if rule_key in rules_dict:
                    c = rules_dict[rule_key]
                    c['count'] = c.get('count', 1) + 1
                    c['output'] = (c['output']*(c['count']-1)+y_train.loc[index])/c['count']
                else:
                    rules_dict[rule_key] = {**rule, 'output': y_train.loc[index], 'count': 1}
        self.rules = list(rules_dict.values())

    def apply_rules(self, row):
        p = 0
        pu, pl = 0, 0
        w_tot = 1e-6
        wu, wl = 1e-6, 1e-6
        for rule in self.rules:
            if self.fuzzy_type == 'type1':
                w = 1
                for col in self.input_columns:
                    w *= self.fuzzy_vars[col][rule[col]](row[col])
                p += rule['output']*w
                w_tot += w
            else:
                w_u, w_l = 1, 1
                for col in self.input_columns:
                    w_u *= self.fuzzy_vars[col][f'{rule[col]}_upper'](row[col])
                    w_l *= self.fuzzy_vars[col][f'{rule[col]}_lower'](row[col])
                wu += w_u
                wl += w_l
                pu += rule['output'][0]*w_u
                pl += rule['output'][1]*w_l
        if self.fuzzy_type == 'type1':
            return p/w_tot
        else:
            return (pu/wu, pl/wl)

    def train(self, X_train, y_train):
        self.create_fuzzy_sets(X_train, y_train)
        self.create_rules(X_train, y_train)

    def predict(self, X_test):
        preds = []
        for _, r in X_test.iterrows():
            preds.append(self.apply_rules(r))
        return preds


class FunctionApproximator:
    def __init__(self, 
                 input_columns, 
                 output_column, 
                 fuzzy_type='type1',
                 fuzzy_shape='All gaussian', 
                 std_factor=1/3,
                 mode='optimize', 
                 kmeans_dim = 1,
                 kmeans_max = 5, 
                 fiting_mode = 'linear',
                 ridge_alphas : list[float] = [1]):
        self.regressor = FuzzyRegression(
            input_columns=input_columns,
            output_column=output_column,
            std_factor=std_factor,
            fuzzy_type=fuzzy_type,
            fuzzy_shape=fuzzy_shape,
            number_of_fuzzy_sets=mode,
            kmeans_dim = kmeans_dim,
            kmeans_max=kmeans_max
        )
        self.fuzzy_sets = None
        self.input_columns = input_columns
        self.output_column = output_column  
        self.ridge_alpha = ridge_alphas
        self.rules = {}
        self.fiting_mode = fiting_mode
    def fit(self, X, y):
        self.regressor.create_fuzzy_sets(X,y)
        self.fuzzy_sets = self.regressor.fuzzy_vars
        self.cross_rules = self.regressor.cross_rules   
        self.create_ns_t1_second_order_rules(X,y)
    
    def predict(self, X):
        y_pred = []
        for _, row in X.iterrows():
            y = 0
            w = 0
            for col in self.input_columns:
                for key, rule in self.rules.items():
                    if key[0] == col:
                        rule_type, coeffs = rule
                        if rule_type == 'linear':
                            y1 = coeffs[0] * row[col] + coeffs[1]
                        elif rule_type == 'quadratic':
                            y1 = coeffs[0] * row[col]**2 + coeffs[1] * row[col] + coeffs[2]
                        elif rule_type == 'cubic':
                            y1 = coeffs[0] * row[col]**3 + coeffs[1] * row[col]**2 + coeffs[2] * row[col] + coeffs[3]
                        elif rule_type == 'exp':
                            y1 = coeffs[0] * np.exp(row[col]) + coeffs[1] * row[col] + coeffs[2]
                        
                        weight = self.fuzzy_sets[col][key[1]](row[col])
                        y += y1 * weight
                        w += weight
            y_pred.append(y/w if w > 0 else 0)
        return np.array(y_pred)

    def create_ns_t1_second_order_rules(self, X, y):
        for col in self.input_columns:
            partition_points = sorted(self.cross_rules[col])
            
            partitions = []
            # print(len(partition_points))
            for i in range(len(partition_points) + 1):
                if i == 0:
                    mask = X[col] <= partition_points[i]
                elif i == len(partition_points):
                    mask = X[col] > partition_points[i-1]
                else:
                    mask = (X[col] > partition_points[i-1]) & (X[col] <= partition_points[i])
                    
                partition_data = X[mask]
                partition_y = y[mask]
                
                max_membership = 0
                dominant_set = None
                for set_name in self.fuzzy_sets[col]:
                    membership = partition_data[col].apply(self.fuzzy_sets[col][set_name]).mean()
                    if membership > max_membership:
                        max_membership = membership
                        dominant_set = set_name
                
                if len(partition_data) > 0:
                    if self.ridge_alpha == 'optimize':
                        ridge_linear = Ridge(alpha=self.ridge_alpha[0])
                        ridge_quadratic = Ridge(alpha=self.ridge_alpha[1])
                        ridge_cubic = Ridge(alpha=self.ridge_alpha[2])
                        ridge_exp = Ridge(alpha=self.ridge_alpha[3])
                    else:
                        ridge_linear = Ridge(alpha=self.ridge_alpha[0])
                        ridge_quadratic = Ridge(alpha=self.ridge_alpha[0])
                        ridge_cubic = Ridge(alpha=self.ridge_alpha[0])
                        ridge_exp = Ridge(alpha=self.ridge_alpha[0])

                    best_score = float('-inf')
                    best_coeffs = None
                    
                    if self.fiting_mode == 'linear':
                        X_poly = np.vstack([partition_data[col], np.ones(len(partition_data))]).T
                        ridge_linear.fit(X_poly, partition_y)
                        coeffs = np.array([ridge_linear.coef_[0], ridge_linear.intercept_])
                        self.rules[(col, dominant_set)] = ('linear', coeffs)
                        
                    elif self.fiting_mode == 'quadratic':
                        X_poly = np.vstack([partition_data[col]**2, partition_data[col], np.ones(len(partition_data))]).T
                        ridge_quadratic.fit(X_poly, partition_y)
                        coeffs = np.append(ridge_quadratic.coef_, ridge_quadratic.intercept_)
                        self.rules[(col, dominant_set)] = ('quadratic', coeffs)
                        
                    elif self.fiting_mode == 'cubic':
                        X_poly = np.vstack([partition_data[col]**3, partition_data[col]**2, partition_data[col], np.ones(len(partition_data))]).T
                        ridge_cubic.fit(X_poly, partition_y)
                        coeffs = np.append(ridge_cubic.coef_, ridge_cubic.intercept_)
                        self.rules[(col, dominant_set)] = ('cubic', coeffs)

                    elif self.fiting_mode == 'exponential':
                        X_poly = np.vstack([np.exp(partition_data[col]), partition_data[col], np.ones(len(partition_data))]).T
                        ridge_exp.fit(X_poly, partition_y)
                        coeffs = np.append(ridge_exp.coef_, ridge_exp.intercept_)
                        self.rules[(col, dominant_set)] = ('exp', coeffs)
                        
                    elif self.fiting_mode == 'both':
                        X_poly = np.vstack([partition_data[col], np.ones(len(partition_data))]).T
                        ridge_linear.fit(X_poly, partition_y)
                        y_pred = ridge_linear.predict(X_poly)
                        linear_score = mean_squared_error(partition_y, y_pred)
                        linear_coeffs = np.array([ridge_linear.coef_[0], ridge_linear.intercept_])
                        
                        X_poly = np.vstack([partition_data[col]**2, partition_data[col], np.ones(len(partition_data))]).T
                        ridge_quadratic.fit(X_poly, partition_y)
                        y_pred = ridge_quadratic.predict(X_poly)
                        quadratic_score = mean_squared_error(partition_y, y_pred)
                        quadratic_coeffs = np.append(ridge_quadratic.coef_, ridge_quadratic.intercept_)
                        
                        X_poly = np.vstack([partition_data[col]**3, partition_data[col]**2, partition_data[col], np.ones(len(partition_data))]).T
                        ridge_cubic.fit(X_poly, partition_y)
                        y_pred = ridge_cubic.predict(X_poly)
                        cubic_score = mean_squared_error(partition_y, y_pred)
                        cubic_coeffs = np.append(ridge_cubic.coef_, ridge_cubic.intercept_)

                        X_poly = np.vstack([np.exp(partition_data[col]), partition_data[col], np.ones(len(partition_data))]).T
                        ridge_exp.fit(X_poly, partition_y)
                        y_pred = ridge_exp.predict(X_poly)
                        exp_score = mean_squared_error(partition_y, y_pred)
                        exp_coeffs = np.append(ridge_exp.coef_, ridge_exp.intercept_)
                        
                        scores = [linear_score, quadratic_score, cubic_score, exp_score]
                        coeffs = [
                            ('linear', linear_coeffs),
                            ('quadratic', quadratic_coeffs), 
                            ('cubic', cubic_coeffs),
                            ('exp', exp_coeffs)
                        ]
                        best_idx = np.argmax(scores)
                        self.rules[(col, dominant_set)] = coeffs[best_idx]
    
    def write_rules(self):
        # print("Number of fuzzy sets:",len(self.fuzzy_sets['x'].keys()))
        with open('rules.txt', 'w') as f:
            for key, rule in self.rules.items():
                model_type, coeffs = rule
                if model_type == 'exp':
                    string_rule = f"{np.round(coeffs[0], 4)}*exp(x) + {np.round(coeffs[1], 4)}*x + {np.round(coeffs[2], 4)}"
                elif model_type == 'linear':
                    string_rule = f"{np.round(coeffs[0], 4)}*x + {np.round(coeffs[1], 4)}"
                elif model_type == 'quadratic':
                    string_rule = f"{np.round(coeffs[0], 4)}*x^2 + {np.round(coeffs[1], 4)}*x + {np.round(coeffs[2], 4)}"
                elif model_type == 'cubic':
                    string_rule = f"{np.round(coeffs[0], 4)}*x^3 + {np.round(coeffs[1], 4)}*x^2 + {np.round(coeffs[2], 4)}*x + {np.round(coeffs[3], 4)}"
                f.write(f"IF X is {key[1]} THEN G(x) is {string_rule}\n")

