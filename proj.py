import sys
import ast
import re
import math
from minizinc import Instance, Model, Solver
from datetime import timedelta
from time import time

# Constants
SOLVER = "chuffed"
TIME_LIMIT = timedelta(seconds=20)
TIME_LIMIT_MINIZINC = timedelta(seconds=10)


def parse_input(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    num_tests = int(lines[0].split()[5])
    num_machines = int(lines[1].split()[5])
    num_resources = int(lines[2].split()[5])
    durations = []
    machine_eligible = []
    required_resources = []
    min_test_duration = int(lines[3:num_tests+3][0].split(", ")[1])
    total_resource_time = [0 for i in range(num_resources)] # total time that each resource is used

    i = 1
    for line in lines[3:num_tests+3]:

        parts = line.split(", ")

        duration = int(parts[1])
        if duration < min_test_duration:
            min_test_duration = duration
        durations.append(duration)
        if len(ast.literal_eval(parts[2])) < 1:
            machines = [True for i in range(num_machines)]
            machine_eligible.append(machines)
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
                    # change the total time that each resource is used
                    total_resource_time[i] += duration
            required_resources.append(resources_bool)
        i += 1
        
    return num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, min_test_duration

def write_output(output_file, makespan, start_times, assigned_machines, num_machines, num_tests, required_resources):

    machine_tests = []
    for i in range(num_machines):
        tests = [(f"'t{t+1}'", start_times[t]) for t in range(num_tests) if assigned_machines[t] == i + 1]
        tests.sort(key=lambda x: x[1])
        machine_tests.append(tests)

    with open(output_file, 'w') as file:
        file.write(f"% Makespan : {makespan}\n")
        for i, tests in enumerate(machine_tests):
            file.write(f"machine('m{i+1}', {len(tests)}, [")
            for j, (test, start_time) in enumerate(tests):
                test_int = int(test[2:-1]) - 1
                file.write(f"({test}, {start_time}")
                if any(required_resources[test_int]):
                    file.write(", [")
                    for k, resource in enumerate(required_resources[test_int]):
                        if resource:
                            file.write(f"'r{k+1}'")
                            if k != len(required_resources[test_int]) - 1:
                                file.write(", ")
                    file.write("]")
                file.write(")")
                
                if j != len(tests) - 1:
                    file.write(", ")
            file.write("])\n")

def find_max_makespan(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources):

    # create a list of the machines by least problematic to most problematic
    machines_estimated_time = [0 for i in range(num_machines)]
    for i in range(num_tests):
        count_available_machines = machine_eligible[i].count(True)
        for m in range(num_machines):
            if machine_eligible[i][m]:
                machines_estimated_time[m] += (durations[i] / count_available_machines)
    
    machines_ordered = [i+1 for i in range(num_machines)]
    machines_ordered.sort(key=lambda x: machines_estimated_time[x-1])


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

    def find_smallest_gap_for_test(test, machines_ordered, max_end_time):
        # find the smallest gap in the machine_usage matrix for the test
        smallest_gap = sum_durations
        gap_type = 1 # 0 if the gap is in the middle of other tests, 1 if the gap is at the end of the tests
        start_time = sum_durations
        machine = -1
        resources_required = [r for r in range(num_resources) if required_resources[test][r]]
        for m in machines_ordered:
            if not machine_eligible[test][m]:
                continue
            t = 0
            while t <= max_end_time:
                if gap_type == 0:
                    break
                # skip if the machine already has a test running or at least one of the resources required for this test is being used
                if machine_usage[m][t] or any([resource_usage[r][t] for r in resources_required]):
                    t += 1
                    continue
                # find the gap
                gap = 0
                while t < max_end_time and not machine_usage[m][t] and not any([resource_usage[r][t] for r in resources_required]):
                    gap += 1
                    t += 1
                if t == max_end_time:
                    if gap_type == 1 and t - gap < start_time:
                        start_time = t - gap
                        machine = m
                        max_end_time = start_time + durations[test]
                    break
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

    start_times = [0 for i in range(num_tests)]
    assigned_machines = [0 for i in range(num_tests)]
    ordered_machines = [i - 1 for i in machines_ordered]
    max_end_time = 0
    for test in tests:
        machine, start_time = find_smallest_gap_for_test(test, ordered_machines, max_end_time)
        if start_time + durations[test] > max_end_time:
            max_end_time = start_time + durations[test]
        introduce_test_in_machine(test, machine, start_time)
        start_times[test] = start_time
        assigned_machines[test] = machine + 1


    # find the maximum makespan
    makespan = 0
    for m in range(num_machines):
        for t in range(sum_durations):
            if machine_usage[m][t]:
                makespan = max(makespan, t + 1)
    
    # return the start times for all tests and machines assigned to each test

    return makespan, start_times, assigned_machines

def find_min_makespan(num_tests, num_machines, num_resources, durations, required_machines, required_resources):

    # for all resources, sum the durations of the tests that require that resource
    resource_durations = [0 for i in range(num_resources)]
    for i in range(num_tests):
        for j in range(num_resources):
            if required_resources[i][j]:
                resource_durations[j] += durations[i]

    sum_durations = sum(durations)
    #divide the sum of the durations by the number of machines
    avg_duration = math.ceil(sum_durations / num_machines)
    max_resources = max(resource_durations)

    min_makespan = max(avg_duration, max_resources)

    # order tests by the number of machines they can run on, tests that can run in less machines first
    tests_ordered_machines = [i+1 for i in range(num_tests) if not all(required_machines[i])]
    tests_ordered_machines.sort(key=lambda x: required_machines[x-1].count(True))

    # order tests by their duration, tests that take more time first
    tests_ordered_no_machines = [i+1 for i in range(num_tests) if all(required_machines[i])]
    tests_ordered_no_machines.sort(key=lambda x: durations[x-1])

    tests_ordered = tests_ordered_machines + tests_ordered_no_machines


    #min_makespan_aux = find_min_makespan_aux(num_tests, num_machines, durations, required_machines, min_makespan, max_makespan, tests_ordered)
    min_makespan_aux = min_makespan
    if avg_duration > max_resources and avg_duration >= min_makespan_aux:
        print("AVERAGE DURATION IS THE BEST OPTION")
    elif max_resources > avg_duration and max_resources >= min_makespan_aux:
        print("MAX RESOURCES IS THE BEST OPTION")
    else:
        print("MINIZINC IS THE WAY TO GO")
    print("max_resources", max_resources)
    print("avg_duration", avg_duration)
    print("min_makespan_aux", min_makespan_aux)

    return max(min_makespan, min_makespan_aux), max_resources


def assign_obvious_machines(machine_eligible, num_tests):
    assigned_machines = [0 for i in range(num_tests)]
    for i in range(num_tests):
        if machine_eligible[i].count(True) == 1:
            assigned_machines[i] = machine_eligible[i].index(True) + 1
    return assigned_machines

def get_resource_usage(num_tests, num_resources, required_resources):
    """the following code is used to obtain lists of tests that use exactly the same resource,
    with also an extra list which serves as an index for each resource, to allow the minizinc model
    to access the tests that use the same resources in a more efficient way"""

    # required_resources_updated is a new required_resources with everything the same, but different pointer
    # we will use it to remove unused resources
    required_resources_updated = [[required_resources[i][j] for j in range(num_resources)] for i in range(num_tests)]
    num_resources_updated = num_resources
    resource_usage = [0 for i in range(num_resources_updated)]
    for i in range(num_tests):
        for j in range(num_resources):
            if required_resources_updated[i][j]:
                resource_usage[j] += 1

    # start from the end of the list
    index_list = []
    for index, res in enumerate(resource_usage):
        if res == 0:
            index_list.append(index)
            num_resources_updated -= 1
            # remove the column of the required resources matrix
    resource_usage = [res for res in resource_usage if res > 0]
    for i in range(num_tests):
        for index in index_list[::-1]:
            required_resources_updated[i].pop(index)
    resource_splits = []
    resource_splits.append(1)
    for i in range(num_resources_updated):
        resource_splits.append(resource_usage[i] + resource_splits[i])

    tests_ordered_by_resource_ids = []
    for r in range(num_resources_updated):
        for t in range(num_tests):
            if required_resources_updated[t][r]:
                tests_ordered_by_resource_ids.append(t+1)
    return resource_splits, tests_ordered_by_resource_ids, required_resources_updated, num_resources_updated


def get_identical_resource_tests(num_tests, num_resources, required_resources):
    """the following code is used to create the list of tests that use exactly the same resources,
    for quicker access to these tests inside the minizinc model"""
    resource_dict = {}
    for t in range(num_tests):
        res_tuple = tuple([i + 1 for i in range(num_resources) if required_resources[t][i]])
        resource_dict[res_tuple] = resource_dict.get(res_tuple, ()) + (t + 1,)
    tests_with_same_resources = []
    tests_with_same_resources_lists = []
    tests_with_same_resources_splits = []
    tests_with_same_resources_splits.append(1)
    for key in resource_dict:
        tests_with_same_resources_lists += [resource_dict[key]]
        if len(resource_dict[key]) > 1 and len(key) > 0:
            tests_with_same_resources += [el for el in resource_dict[key]]
            tests_with_same_resources_splits.append(len(resource_dict[key]) + tests_with_same_resources_splits[-1])
    if () in resource_dict:
        tests_with_same_resources += [el for el in resource_dict[()]]
        tests_with_same_resources_splits.append(len(resource_dict[()]) + tests_with_same_resources_splits[-1])
    num_tests_with_same_resources_splits = len(tests_with_same_resources_splits) - 1
    num_tests_with_same_resources = len(tests_with_same_resources)

    return tests_with_same_resources, tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits


def check_equal_tests(num_tests, num_machines, num_resources, machine_eligible, required_resources, durations):
    count = 0
    equal_tests = []
    for i in range(num_tests):
        for j in range(i+1, num_tests):
            if all([machine_eligible[i][m] == machine_eligible[j][m] for m in range(num_machines)]) and all([required_resources[i][r] == required_resources[j][r] for r in range(num_resources)]) and durations[i] == durations[j]:
                print(f"test {i+1} and test {j+1} are equal")
                equal_tests.append((i+1, j+1))
                count += 1
    print(f"Number of equal tests: {count}")
    return equal_tests, len(equal_tests)


def check_equal_machines(num_tests, num_machines, machine_eligible):
    
    parallelizable_machines = [] # list of lists

    parallelizable_machines.append([i for i in range(num_machines)])

    for i in range(num_tests):
        new_parallelizable_machines = []
        for parallel_set in parallelizable_machines:
            can_run_list = []
            cant_run_list = []
            for m in parallel_set:
                if machine_eligible[i][m]:
                    can_run_list.append(m)
                else:
                    cant_run_list.append(m)
            if len(can_run_list) > 0:
                new_parallelizable_machines.append(can_run_list)
            if len(cant_run_list) > 0:
                new_parallelizable_machines.append(cant_run_list)
        parallelizable_machines = new_parallelizable_machines

    i = 0
    equal_machines = []
    for parallel_set in parallelizable_machines:
        if len(parallel_set) > 1:
            i += 1
            print(f"parallelizable machines {i}: ", [k+1 for k in parallel_set])
    for parallel_set in parallelizable_machines:
        if len(parallel_set) > 1:
            for i in range(len(parallel_set) - 1):
                equal_machines.append((parallel_set[i]+1, parallel_set[i+1]+1))
    return equal_machines, len(equal_machines)

def create_and_run_minizinc_model(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, target_makespan,
                                  min_test_duration, hardcoded_machines, initial_assigned_machines, initial_start_times,
                                  resource_splits, total_resources_num, tests_ordered_by_resource_ids, tests_with_same_resources,
                                  tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits,
                                  eq_tests, num_eq_tests, eq_machines, num_eq_machines, hardcoded_start_times):
    
    model = Model("proj_satisfy.mzn")
    solver = Solver.lookup(SOLVER)

    # Create an instance of the model
    instance = Instance(solver, model)
    instance["num_tests"] = num_tests
    instance["num_machines"] = num_machines
    instance["num_resources"] = num_resources
    instance["durations"] = durations
    instance["required_machines"] = machine_eligible
    instance["required_resources"] = required_resources
    instance["target_makespan"] = target_makespan
    instance["min_test_duration"] = min_test_duration
    instance["hardcoded_machines"] = hardcoded_machines
    instance["initial_assigned_machines"] = initial_assigned_machines
    instance["initial_start_times"] = initial_start_times
    instance["resource_start_times"] =  hardcoded_start_times

    instance["resource_splits"] = resource_splits
    instance["total_resources_num"] = total_resources_num
    instance["tests_ordered_by_resource_ids"] = tests_ordered_by_resource_ids

    instance["tests_with_same_resources"] = tests_with_same_resources
    instance["tests_with_same_resources_splits"] = tests_with_same_resources_splits
    instance["num_tests_with_same_resources"] = num_tests_with_same_resources
    instance["num_tests_with_same_resources_splits"] = num_tests_with_same_resources_splits

    instance["num_eq_tests"] = num_eq_tests
    instance["eq_tests"] = eq_tests
    instance["num_eq_machines"] = num_eq_machines
    instance["eq_machines"] = eq_machines

    print("Currently tring to solve the model with makespan: ", target_makespan)
    start_time = time()
    result = instance.solve()
    end_time = time()
    print(f"Time taken: {end_time - start_time}")

    if result.status.has_solution():
        print(result)
        write_output(output_file, target_makespan, result["start_times"], result["assigned_machines"], num_machines, num_tests, required_resources)
        print(f"Current best solution: {target_makespan}")
        return True
    else:
        print(f"Status: {result.status}")  # This will give more details on what went wrong"""
        return False


def run_alternative_model(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, target_makespan,
                                    min_test_duration, hardcoded_machines, initial_assigned_machines, initial_start_times,
                                    resource_splits, total_resources_num, tests_ordered_by_resource_ids, tests_with_same_resources,
                                    tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits,
                                    eq_tests, num_eq_tests, eq_machines, num_eq_machines):
        
        model = Model("proj_satisfy_rec.mzn")
        solver = Solver.lookup(SOLVER)
    
        # Create an instance of the model
        instance = Instance(solver, model)
        instance["num_tests"] = num_tests
        instance["num_machines"] = num_machines
        instance["num_resources"] = num_resources
        instance["durations"] = durations
        instance["required_machines"] = machine_eligible
        instance["required_resources"] = required_resources
        instance["target_makespan"] = target_makespan
        instance["min_test_duration"] = min_test_duration
        instance["hardcoded_machines"] = hardcoded_machines
        instance["initial_assigned_machines"] = initial_assigned_machines
        instance["initial_start_times"] = initial_start_times
    
        instance["resource_splits"] = resource_splits
        instance["total_resources_num"] = total_resources_num
        instance["tests_ordered_by_resource_ids"] = tests_ordered_by_resource_ids
    
        instance["tests_with_same_resources"] = tests_with_same_resources
        instance["tests_with_same_resources_splits"] = tests_with_same_resources_splits
        instance["num_tests_with_same_resources"] = num_tests_with_same_resources
        instance["num_tests_with_same_resources_splits"] = num_tests_with_same_resources_splits
    
        instance["num_eq_tests"] = num_eq_tests
        instance["eq_tests"] = eq_tests
        instance["num_eq_machines"] = num_eq_machines
        instance["eq_machines"] = eq_machines
    
        print("Currently tring to solve the alternative model")
        start_time = time()
        result = instance.solve()
        end_time = time()
        print(f"Time taken: {end_time - start_time}")
    
        if result.status.has_solution():
            print(result)
            write_output(output_file, target_makespan, result["start_times"], result["assigned_machines"], num_machines, num_tests, required_resources)
            print(f"Current Best solution: {target_makespan}")
            return True
        else:
            print(f"Status: {result.status}")  # This will give more details on what went wrong"""
            return False

def binary_search(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, min_makespan, max_makespan,
                                  min_test_duration, hardcoded_machines, initial_assigned_machines, initial_start_times,
                                  resource_splits, total_resources_num, tests_ordered_by_resource_ids, tests_with_same_resources,
                                  tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits,
                                  eq_tests, num_eq_tests, eq_machines, num_eq_machines, hardcoded_start_times):
    
    use_alternative_model = False
    # start by running the minizinc model with the max makespan, to get a solution
    found_solution = create_and_run_minizinc_model(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, max_makespan,
                                    min_test_duration, hardcoded_machines, initial_assigned_machines, initial_start_times,
                                    resource_splits, total_resources_num, tests_ordered_by_resource_ids, tests_with_same_resources,
                                    tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits,
                                    eq_tests, num_eq_tests, eq_machines, num_eq_machines, hardcoded_start_times)
    if found_solution:
        best_result = max_makespan
    else:
        best_result = -1
        use_alternative_model = True
        found_solution = run_alternative_model(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, max_makespan,
                                  min_test_duration, hardcoded_machines, initial_assigned_machines, initial_start_times,
                                  resource_splits, total_resources_num, tests_ordered_by_resource_ids, tests_with_same_resources,
                                  tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits,
                                  eq_tests, num_eq_tests, eq_machines, num_eq_machines)
        if found_solution:
            best_result = max_makespan
        else:
            print("No solution found for the max makespan")
            return

    # start the binary search
    left = min_makespan
    right = max_makespan
    while left < right:
        mid = (left + right) // 2
        if use_alternative_model:
            found_solution = run_alternative_model(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, mid,
                                  min_test_duration, hardcoded_machines, initial_assigned_machines, initial_start_times,
                                  resource_splits, total_resources_num, tests_ordered_by_resource_ids, tests_with_same_resources,
                                  tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits,
                                  eq_tests, num_eq_tests, eq_machines, num_eq_machines)
        else:
            found_solution = create_and_run_minizinc_model(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, mid,
                                    min_test_duration, hardcoded_machines, initial_assigned_machines, initial_start_times,
                                    resource_splits, total_resources_num, tests_ordered_by_resource_ids, tests_with_same_resources,
                                    tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits,
                                    eq_tests, num_eq_tests, eq_machines, num_eq_machines, hardcoded_start_times)
        if found_solution:
            right = mid
            best_result = mid
        else:
            left = mid + 1
    if best_result != left:
        if use_alternative_model:
            found_solution = run_alternative_model(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, left,
                                    min_test_duration, hardcoded_machines, initial_assigned_machines, initial_start_times,
                                    resource_splits, total_resources_num, tests_ordered_by_resource_ids, tests_with_same_resources,
                                    tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits,
                                    eq_tests, num_eq_tests, eq_machines, num_eq_machines)
        else:
            found_solution = create_and_run_minizinc_model(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, left,
                                        min_test_duration, hardcoded_machines, initial_assigned_machines, initial_start_times,
                                        resource_splits, total_resources_num, tests_ordered_by_resource_ids, tests_with_same_resources,
                                        tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits,
                                        eq_tests, num_eq_tests, eq_machines, num_eq_machines, hardcoded_start_times)
        if found_solution:
            best_result = left
    print("Final solution found: ", best_result)

def find_start_time_resources(num_tests, num_resources, durations, required_resources, min_makespan, max_makespan):
    """runs an initial instance of minizinc to find the start times of tests that use resources"""

    
    """create a dictionary with the key being a tuple with all the resources used: each key will have two values,
    the first is a list with the ids of all the tests with those same resources, the second is the total duration of those tests"""
    resource_dict = {}
    for t in range(num_tests):
        res_tuple = tuple([i + 1 for i in range(num_resources) if required_resources[t][i]])
        if len(res_tuple) == 0:
            continue
        resource_dict[res_tuple] = resource_dict.get(res_tuple, ([], 0))
        resource_dict[res_tuple][0].append(t + 1)
        resource_dict[res_tuple] = (resource_dict[res_tuple][0], resource_dict[res_tuple][1] + durations[t])
    
    """create a new required_resources table, but using the keys as being only one test"""
    required_resources_groups = []
    for key in resource_dict:
        new_required_resources = []
        for r in range(num_resources):
            if r+1 in key:
                new_required_resources.append(True)
            else:
                new_required_resources.append(False)
        required_resources_groups.append(new_required_resources)
    num_tests_groups = len(required_resources_groups)
    durations_groups = [resource_dict[key][1] for key in resource_dict]
            
    priority_tests = []
    changes = True
    while changes:
        changes = False
        for t in range(len(resource_dict)):
            if t + 1 in priority_tests:
                continue
            has_more_priority = True
            for t2 in range(len(resource_dict)):
                if t == t2 or t2 + 1 in priority_tests:
                    continue
                has_all_resources = True
                for r in range(num_resources):
                    if (not required_resources_groups[t][r]) and required_resources_groups[t2][r]:
                        has_all_resources = False
                        break
                if not has_all_resources:
                    has_more_priority = False
                    break
            if has_more_priority:
                priority_tests.append(t + 1)
                changes = True

    start_times = []
    resource_dict_list = list(resource_dict)
    for el in priority_tests:
        if len(start_times) == 0:
            start_times.append(0)
        else:
            start_times.append(start_times[-1] + resource_dict[resource_dict_list[previous_el - 1]][1])
        previous_el = el
    hardcoded_start_times = []
    for i in range(num_tests_groups):
        if i+1 in priority_tests:
            hardcoded_start_times.append(start_times[priority_tests.index(i+1)])
        else:
            hardcoded_start_times.append(-1)
    resource_splits, tests_ordered_by_resource_ids, _, num_updated_resources = get_resource_usage(
        num_tests_groups, num_resources, required_resources_groups)
    



    



    model = Model("proj_start_times.mzn")
    solver = Solver.lookup(SOLVER)

    # Create an instance of the model
    instance = Instance(solver, model)
    instance["num_tests"] = num_tests_groups
    instance["num_resources"] = num_updated_resources
    instance["durations"] = durations_groups
    instance["required_resources"] = required_resources_groups
    instance["min_makespan"] = min_makespan
    instance["max_makespan"] = max_makespan
    instance["resource_splits"] = resource_splits
    instance["total_resources_num"] = len(tests_ordered_by_resource_ids)
    instance["tests_ordered_by_resource_ids"] = tests_ordered_by_resource_ids
    instance["hardcoded_start_times"] = hardcoded_start_times

    print("Currently tring to solve the model to find start times")
    start_time = time()
    result = instance.solve()
    end_time = time()
    print(f"Time taken: {end_time - start_time}")

    if result.status.has_solution():
        print(result)
        objective = result["objective"]
        start_times = result["start_times"]

        """now go to the resource_dict and compute a new start time for each test, based on the start times of the groups"""
        new_start_times = [-1 for i in range(num_tests)]
        i = 0
        for key in resource_dict:
            start_time_group =  start_times[i]
            start_time_list_group = []
            for test in resource_dict[key][0]:
                start_time_list_group.append(start_time_group)
                start_time_group += durations[test - 1]
            for j in range(len(resource_dict[key][0])):
                new_start_times[resource_dict[key][0][j] - 1] = start_time_list_group[j]
            i += 1
        return objective, new_start_times

    else:
        print(f"Status: {result.status}")  # This will give more details on what went wrong"""
        return None
    

                    


def main(input_file, output_file):


    # Parse the input file
    num_tests, num_machines, num_resources, durations, machine_eligible, required_resources, min_test_duration = parse_input(input_file)

    eq_tests, len_eq_tests = check_equal_tests(num_tests, num_machines, num_resources, machine_eligible, required_resources, durations)
    eq_machines, len_eq_machines = check_equal_machines(num_tests, num_machines, machine_eligible)

    max_makespan, start_times, assigned_machines = find_max_makespan(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources)
    min_makespan, min_makespan_resources = find_min_makespan(num_tests, num_machines, num_resources, durations, machine_eligible, required_resources)
    print('min_makespan', min_makespan)
    print('max_makespan', max_makespan)

    # unmodifiable machines
    hardcoded_machines = assign_obvious_machines(machine_eligible, num_tests)

    # obtain the list of tests that share one resource, plus list to allow quicker access for each resource
    resource_splits, tests_ordered_by_resource_ids, required_resources_updated, num_resources_updated = get_resource_usage(num_tests, num_resources, required_resources)
    # obtain the list of tests that share exactly the same resources, plus a list to allow quicker access for each group of tests
    tests_with_same_resources, tests_with_same_resources_splits, num_tests_with_same_resources, num_tests_with_same_resources_splits = get_identical_resource_tests(num_tests, num_resources, required_resources)

    new_min_makespan, start_times = find_start_time_resources(num_tests, num_resources_updated, durations, required_resources_updated, min_makespan_resources, max_makespan)
    if new_min_makespan is not None:
        if new_min_makespan > min_makespan:
            min_makespan = new_min_makespan
    binary_search(num_tests=num_tests, num_machines=num_machines, num_resources=num_resources_updated, durations=durations, machine_eligible=machine_eligible, required_resources=required_resources_updated, min_makespan=min_makespan, max_makespan=max_makespan,
                                  min_test_duration=min_test_duration, hardcoded_machines=hardcoded_machines, initial_assigned_machines=assigned_machines, initial_start_times=start_times,
                                  resource_splits=resource_splits, total_resources_num=len(tests_ordered_by_resource_ids), tests_ordered_by_resource_ids=tests_ordered_by_resource_ids, tests_with_same_resources=tests_with_same_resources,
                                  tests_with_same_resources_splits=tests_with_same_resources_splits, num_tests_with_same_resources=num_tests_with_same_resources, num_tests_with_same_resources_splits=num_tests_with_same_resources_splits,
                                  eq_tests=eq_tests, num_eq_tests=len_eq_tests, eq_machines=eq_machines, num_eq_machines=len_eq_machines, hardcoded_start_times=start_times) 


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    print(input_file, output_file)
    main(input_file, output_file)
