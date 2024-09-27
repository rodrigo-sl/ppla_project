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
            machines = [int(i)+1 for i in range(num_machines)]
            machine_eligible.append(machines)
            #print('machines', machines)
        else:
            machines_str = re.sub(r"[\[\]'m]", "", parts[2])
            machines = [int(machine) for machine in machines_str.split(',')]
            machine_eligible.append(machines)
            #print('machines', machines)
            
        parts[3] = parts[3].replace(")", "").replace("\n", "")

        if len(ast.literal_eval(parts[3])) < 1:
            resources = []
            required_resources.append(resources)
            #print('resources', resources)
        else:
            resources_str = re.sub(r"[\[\]'r]", "", parts[3])
            resources = [int(resource) for resource in resources_str.split(',')]
            required_resources.append(resources)
            #print('resources', resources)

        i += 1
    
    return num_tests, num_machines, num_resources, tests, durations, machine_eligible, required_resources


def write_output(output_file, makespan, schedule):
    with open(output_file, 'w') as file:
        file.write(f"% Makespan : {makespan}\n")
        for machine, tests in schedule.items():
            file.write(f"machine( '{machine}', {len(tests)}, {tests} )\n")

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

    # Write the results to output file
    with open(output_file, 'w') as f:
        f.write(result["output"])

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
