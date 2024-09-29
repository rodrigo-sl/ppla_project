import sys
import ast
import re
from minizinc import Instance, Model, Solver

def parse_input(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    tests = []
    num_tests = int(lines[0].split()[5])
    num_machines = int(lines[1].split()[5])
    num_resources = int(lines[2].split()[5])
    durations = []
    machine_eligible = []
    required_resources = []

    i = 1
    for line in lines[3:num_tests+3]:

        #print('test', i)

        parts = line.split(", ")
        #print(parts)

        duration = int(parts[1])
        durations.append(duration)
        #print('duration', duration)
        if len(ast.literal_eval(parts[2])) < 1:
            machines = [True for i in range(num_machines)]
            machine_eligible.append(machines)
            #print('machines', machines)
        else:
            machines_str = re.sub(r"[\[\]'m]", "", parts[2])
            machines = [int(machine) for machine in machines_str.split(',')]
            machines_bool = []
            for i in range(num_machines):
                if i+1 not in machines:
                    machines_bool.append(False)
                else:
                    machines_bool.append(True)
            machine_eligible.append(machines_bool)
            #print('machines', machines)
            
        parts[3] = parts[3].replace(")", "").replace("\n", "")

        if len(ast.literal_eval(parts[3])) < 1:
            resources = [False for i in range(num_resources)]
            required_resources.append(resources)
            #print('resources', resources)
        else:
            resources_str = re.sub(r"[\[\]'r]", "", parts[3])
            resources = [int(resource) for resource in resources_str.split(',')]
            resources_bool = []
            for i in range(num_resources):
                if i+1 not in resources:
                    resources_bool.append(False)
                else:
                    resources_bool.append(True)
            required_resources.append(resources_bool)
            #print('resources', resources)

        i += 1
    
    return num_tests, num_machines, num_resources, tests, durations, machine_eligible, required_resources


def write_output(output_file, makespan, start_times, assigned_machines, num_machines, num_tests):

    machine_tests = []
    for i in range(num_machines):
        tests = [(f"t{t+1}", start_times[t]) for t in range(num_tests) if assigned_machines[t] == i + 1]
        tests.sort(key=lambda x: x[1])
        machine_tests.append(tests)

    with open(output_file, 'w') as file:
        file.write(f"% Makespan : {makespan}\n")
        for i, tests in enumerate(machine_tests):
            file.write(f"machine(m{i+1}, {len(tests)}, [")
            for j, (test, start_time) in enumerate(tests):
                file.write(f"({test}, {start_time})")
                if j != len(tests) - 1:
                    file.write(", ")
            file.write("])\n")

def main(input_file, output_file):
    # Load the model
    model = Model("proj.mzn")
    solver = Solver.lookup("gecode")  # Or any other solver you are using

    # Parse the input file (as you've done before)
    num_tests, num_machines, num_resources, tests, durations, machine_eligible, required_resources = parse_input(input_file)

    # Create an instance of the model
    instance = Instance(solver, model)
    instance["num_tests"] = num_tests
    instance["num_machines"] = num_machines
    instance["num_resources"] = num_resources
    instance["durations"] = durations
    instance["required_machines"] = machine_eligible
    instance["required_resources"] = required_resources
    print('num_tests', num_tests)
    print('num_machines', num_machines)
    print('num_resources', num_resources)
    print('durations', durations)
    print('required_machines', machine_eligible)
    print('req_resources', required_resources)
    
    # Solve the model
    result = instance.solve()

    # Check if there is a valid solution
    if result.status.has_solution():
        print(result)
        write_output(output_file, result["objective"], result["start_times"], result["assigned_machines"], num_machines, num_tests)
    else:
        print("No solution found or inconsistency in the model.")
        print(f"Status: {result.status}")  # This will give more details on what went wrong


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    print(input_file, output_file)
    main(input_file, output_file)
