import numpy as np
import pandas as pd



# from pyomo.environ import *
# from sklearn.metrics import r2_score

# #A optimality = minimize trace of inverse information matrix
# def a_optimality(X, allowed_num):
#     print("Choosing a subset using A-Optimality...")
#     n_samples, n_features = X.shape
#     print("Creating Model...")
#     model = ConcreteModel()
#     model.x = Var(range(n_samples), within=Binary)
#     model.n_samples = n_samples

#     def restrict_num_samples(model):
#         total_num = sum(model.x[i] for i in range(model.n_samples))
#         return total_num <= allowed_num

#     print("Adding constraints for number of samples")
#     model.select_points = Constraint(rule=restrict_num_samples)

#     def objective_rule(model):
#         print("Calculating objective rule")
#         # Create the information matrix
#         selected_samples = [i for i in range(n_samples) if model.x[i].value == 1]
#         if not selected_samples:
#             return float('inf')  # If no samples are selected, return infinity
        
#         # Calculate the information matrix M(X)
#         M_matrix = np.zeros((n_features, n_features))
#         for i in selected_samples:
#             M_matrix += np.outer(X[i], X[i])  # Add outer product of selected samples
            
#         if np.linalg.det(M_matrix) == 0:
#             return float('inf')  # Prevent division by zero if M is singular
        
#         M_inv = np.linalg.inv(M_matrix)  # Inverse of the information matrix
#         return sum(M_inv[i, i] for i in range(n_features))  # Trace of the inverse

    
#     model.obj = Objective(rule=objective_rule, sense=minimize)

#     # Constraint to select exactly num_samples points
#     print("Setting constraint to choosing n_samples")

#     from IPython import embed; embed()
#     # Solve the model
#     print("Solving the model...")
#     solver = SolverFactory('glpk')
#     results = solver.solve(model, tee=True)

#     return [i for i in range(n_samples) if model.select[i].value == 1]

# #D Optimality = maximize the determinant of the information matrix.
# def d_optimality(X, num_points):
#     n_samples = X.shape[0]
#     model = ConcreteModel()
#     model.select = Var(range(n_samples), within=Binary)

#     #CHANGE TO COST CONSTRAINT
#     model.num_points_constraint = Constraint(expr=sum(model.select[i] for i in range(n_samples)) == num_points)

#     def objective_rule(model):
#         selected_points = [i for i in range(n_samples) if model.select[i].value == 1]
#         if not selected_points:
#             return float('inf')
#         X_selected = X[selected_points, :]
#         info_matrix = X_selected.T @ X_selected
#         return -np.linalg.det(info_matrix + 1e-10 * np.eye(X.shape[1]))  # Negative for maximization

#     model.D_optimality = Objective(rule=objective_rule, sense=minimize)
#     solver = SolverFactory('glpk')
#     solver.solve(model)

#     return [i for i in range(n_samples) if model.select[i].value == 1]

# #E Optimality =  minimize the largest eigenvalue of the inverse information matrix.
# def e_optimality(X, num_points):
#     n_samples = X.shape[0]
#     model = ConcreteModel()
#     model.select = Var(range(n_samples), within=Binary)
#     model.num_points_constraint = Constraint(expr=sum(model.select[i] for i in range(n_samples)) == num_points)

#     def objective_rule(model):
#         selected_points = [i for i in range(n_samples) if model.select[i].value == 1]
#         if not selected_points:
#             return float('inf')
#         X_selected = X[selected_points, :]
#         info_matrix = X_selected.T @ X_selected
#         eigenvalues = np.linalg.eigvals(np.linalg.inv(info_matrix + 1e-10 * np.eye(X.shape[1])))
#         return max(eigenvalues)

#     model.E_optimality = Objective(rule=objective_rule, sense=minimize)
#     solver = SolverFactory('glpk')
#     solver.solve(model)

#     return [i for i in range(n_samples) if model.select[i].value == 1]

# #V-Optimality = minimize the average variance at specific prediction points Y.
# def v_optimality(X, Y, num_points):
#     n_samples = X.shape[0]
#     model = ConcreteModel()
#     model.select = Var(range(n_samples), within=Binary)
#     model.num_points_constraint = Constraint(expr=sum(model.select[i] for i in range(n_samples)) == num_points)

#     def objective_rule(model):
#         selected_points = [i for i in range(n_samples) if model.select[i].value == 1]
#         if not selected_points:
#             return float('inf')
#         X_selected = X[selected_points, :]
#         info_matrix = X_selected.T @ X_selected
#         inv_info_matrix = np.linalg.inv(info_matrix + 1e-10 * np.eye(X.shape[1]))
#         prediction_variances = [np.dot(y.T, np.dot(inv_info_matrix, y)) for y in Y]
#         return sum(prediction_variances) / len(Y)

#     model.V_optimality = Objective(rule=objective_rule, sense=minimize)
#     solver = SolverFactory('glpk')
#     solver.solve(model)

#     return [i for i in range(n_samples) if model.select[i].value == 1]