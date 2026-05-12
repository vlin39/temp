from ortools.linear_solver.python import model_builder


def dietProblem():
    # Diet Problem
    model = model_builder.Model()
    solver = model_builder.Solver("SCIP")

    # Add the variables
    # For Linear Programs with continuous float variables, use model.new_num_var()
    # For Integer Programs with discrete integer variables, use model.new_int_var()
    meat = model.new_num_var(0.0, 1000.0, "meat")
    bread = model.new_num_var(0.0, 1000.0, "bread")

    # Add the constraints
    model.add(100 * meat + 250 * bread >= 500)  # Carbs
    model.add(100 * meat + 50 * bread >= 250)  # Fat
    model.add(150 * meat + 200 * bread >= 600)  # Protein

    # Add the objective
    model.minimize(25 * meat + 15 * bread)

    # Solve the problem
    status = solver.solve(model)
    obj_value = solver.objective_value

    # Make sure to check solver status!!
    if status == model_builder.SolveStatus.OPTIMAL:
        print(f"Number of variables = {model.num_variables}")
        print(f"Number of constraints = {model.num_constraints}")
        print(f"Meat: {solver.value(meat)}")
        print(f"Bread: {solver.value(bread)}")
        print(f"Objective Value: {obj_value}")
    else:
        print("Status: ", status)


if __name__ == "__main__":
    dietProblem()
