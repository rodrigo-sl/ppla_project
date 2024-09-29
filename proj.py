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

def find_max_makespan(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources):

    # compute the sum of all the durations
    sum_durations = sum(durations)

    # create a matrix machine usage that is num_machines x sum_durations. every element is 0 at the start
    machine_usage = [[False for i in range(sum_durations)] for j in range(num_machines)]

    # create a matrix resource usage that is num_resources x sum_durations. every element is 0 at the start
    resource_usage = [[False for i in range(sum_durations)] for j in range(num_resources)]

    def introduce_test_in_machine(test, machine, start_time):
        # introduce the test in the machine_usage matrix
        for i in range(durations[test]):
            machine_usage[machine][start_time + i] = True

        # introduce the test in the resource_usage matrix
        for i in range(num_resources):
            if required_resources[test][i]:
                for j in range(durations[test]):
                    resource_usage[i][start_time + j] = True

    def find_smallest_gap_for_test(test):
        # find the smallest gap in the machine_usage matrix for the test
        smallest_gap = -1
        gap_type = 1 # 0 if the gap is in the middle of other tests, 1 if the gap is at the end of the tests
        start_time = sum_durations
        machine = -1
        for m in range(num_machines):
            if not machine_eligible[test][m]:
                continue
            t = 0
            while t < sum_durations:
                # skip if the machine already has a test running or at least one of the resources required for this test is being used
                if machine_usage[m][t] or any([resource_usage[r][t] for r in range(num_resources) if required_resources[test][r]]):
                    t += 1
                    continue
                # find the gap
                gap = 0
                while t < sum_durations and not machine_usage[m][t] and not any([resource_usage[r][t] for r in range(num_resources) if required_resources[test][r]]):
                    gap += 1
                    t += 1
                if t == sum_durations:
                    if gap >= durations[test] and gap > smallest_gap:
                        smallest_gap = gap
                        start_time = t - gap
                        machine = m
                else:
                    if gap >= durations[test] and (gap_type == 1 or gap < smallest_gap or smallest_gap == -1):
                        smallest_gap = gap
                        start_time = t - gap
                        machine = m
                        gap_type = 0
        return machine, start_time
    
    # for each test, find the smallest gap or the gap with the smallest start time and introduce the test in
    # the machine_usage and resource_usage matrices, starting with tests that need the most resources

    # select only the tests with resource constraints
    tests_with_resources = [i for i in range(num_tests) if any(required_resources[i])]
    tests_with_resources.sort(key=lambda x: sum(required_resources[x]), reverse=True)
    
    # select the tests with machine constraints
    tests_with_machines = [i for i in range(num_tests) if not any(required_resources[i]) and not all(machine_eligible[i])]
    tests_with_machines.sort(key=lambda x: durations[x], reverse=True)

    # select the tests with no constraints
    tests_no_constraints = [i for i in range(num_tests) if not any(required_resources[i]) and all(machine_eligible[i])]
    tests_no_constraints.sort(key=lambda x: durations[x], reverse=True)

    tests = tests_with_resources + tests_with_machines + tests_no_constraints

    for test in tests:
        machine, start_time = find_smallest_gap_for_test(test)
        introduce_test_in_machine(test, machine, start_time)

    # find the maximum makespan
    makespan = 0
    for m in range(num_machines):
        for t in range(sum_durations):
            if machine_usage[m][t]:
                makespan = max(makespan, t + 1)
    
    return makespan

def find_min_makespan(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources):

    # for all resources, sum the durations of the tests that require that resource
    resource_durations = [0 for i in range(num_resources)]
    for i in range(num_tests):
        for j in range(num_resources):
            if required_resources[i][j]:
                resource_durations[j] += durations[i]
    return max(resource_durations)
    


def main(input_file, output_file):
    # Load the model
    model = Model("proj.mzn")
    solver = Solver.lookup("gecode")  # Or any other solver you are using

    # Parse the input file (as you've done before)
    num_tests, num_machines, num_resources, tests, durations, machine_eligible, required_resources = parse_input(input_file)

    min_makespan = find_min_makespan(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources)
    max_makespan = find_max_makespan(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources)
    print('min_makespan', min_makespan)
    print('max_makespan', max_makespan)


    # Create an instance of the model
    instance = Instance(solver, model)
    instance["num_tests"] = num_tests
    instance["num_machines"] = num_machines
    instance["num_resources"] = num_resources
    instance["durations"] = durations
    instance["required_machines"] = machine_eligible
    instance["required_resources"] = required_resources
    instance["min_makespan"] = min_makespan
    instance["max_makespan"] = max_makespan
    """print('num_tests', num_tests)
    print('num_machines', num_machines)
    print('num_resources', num_resources)
    print('durations', durations)
    print('required_machines', machine_eligible)
    print('req_resources', required_resources)"""
    
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
