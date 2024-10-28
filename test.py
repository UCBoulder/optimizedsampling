# Open and read the LP file
with open('/tmp/tmp0gkxafv2.pyomo.lp', 'r') as file:
    lines = file.readlines()

# Initialize lists to hold variables and constraints
variables = []
constraints = []

# Parse the lines for variables and constraints
for line in lines:
    if line.startswith('var '):
        variables.append(line.strip())
    elif line.startswith('s.t. '):  # Change this if your constraints have a different prefix
        constraints.append(line.strip())

# Print out the found variables and constraints
print("Variables:")
for var in variables:
    print(var)

print("\nConstraints:")
for constr in constraints:
    print(constr)
