int: num_tests;
int: num_machines;
int: num_resources;

array[1..num_tests] of int: durations;
array[1..num_tests] of set of int: required_machines;
array[1..num_tests] of set of int: required_resources;

% Decision variables
array[1..num_tests] of var 0..15: start_times; % Start time for each test
array[1..num_tests] of var 1..num_machines: assigned_machines; % Machine assigned to each test
array[1..num_tests, 1..num_resources] of var bool: uses_resources; % Whether each resource is used by a test

% Compute end times for each test
array[1..num_tests] of var int: end_times = [start_times[i] + durations[i] | i in 1..num_tests];

% Constraints
constraint forall(i in 1..num_tests)(
    start_times[i] + durations[i] <= 15  % Ensure the test finishes within max makespan
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
        (uses_resources[i, r] == false \/ uses_resources[j, r] == false) \/
        (start_times[i] + durations[i] <= start_times[j]) \/
        (start_times[j] + durations[j] <= start_times[i])
    )
);

% Minimize makespan (the maximum end time)
var int: makespan = max(end_times);

solve minimize makespan;


