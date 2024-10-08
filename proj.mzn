int: num_tests;
int: num_machines;
int: num_resources;
int: min_makespan;
int: max_makespan;

array[1..num_tests] of int: durations;
array[1..num_tests, 1..num_machines] of bool: required_machines;
array[1..num_tests, 1..num_resources] of bool: required_resources;

% Decision variables
array[1..num_tests] of var int: start_times; % Start time for each test
array[1..num_tests] of var 1..num_machines: assigned_machines; % Machine assigned to each test

% Compute end times for each test
array[1..num_tests] of var int: end_times = [start_times[i] + durations[i] | i in 1..num_tests];

var min_makespan..max_makespan: makespan = max(end_times);

% Constraints

constraint forall(i in 1..num_tests)(
    start_times[i] >= 0  % Ensure the test starts after time 0
);

% the machine assigned to east test must be flagged as true in the required_machines
constraint forall(i in 1..num_tests)(
    required_machines[i, assigned_machines[i]] == true
);

% Prevent overlapping of tests on the same machine
constraint forall(i, j in 1..num_tests where i != j)(
    (assigned_machines[i] != assigned_machines[j]) \/
    (start_times[i] + durations[i] <= start_times[j]) \/
    (start_times[j] + durations[j] <= start_times[i])
);

% Prevent overlapping of resource usage
constraint forall(r in 1..num_resources)(
    forall(i, j in 1..num_tests where i != j)(
        (required_resources[i, r] == false \/ required_resources[j, r] == false) \/
        (start_times[i] + durations[i] <= start_times[j]) \/
        (start_times[j] + durations[j] <= start_times[i])
    )
);

% Minimize makespan (the maximum end time)
solve minimize makespan;
