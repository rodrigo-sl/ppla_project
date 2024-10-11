int: num_tests;
int: num_machines;
int: num_resources;
int: target_makespan;
int: min_test_duration;

array[1..num_tests] of int: durations;
array[1..num_tests, 1..num_machines] of bool: required_machines;
array[1..num_tests, 1..num_resources] of bool: required_resources;

set of int: machines_ordered;  % Specific order of machines
set of int: tests_ordered;  % Specific order of tests
% Decision variables
array[1..num_tests] of var 1..num_machines: assigned_machines; % Machine assigned to each test
array[1..num_tests] of var 0..target_makespan - min_test_duration: start_times; % Start time for each test
% hardcoded arrays
array[1..num_tests] of 0..num_machines: hardcoded_machines;
% warm-start
%array[1..num_tests] of 1..num_machines: initial_assigned_machines;
%array[1..num_tests] of 0..target_makespan - min_test_duration: initial_start_times;

% Compute end times for each test
array[1..num_tests] of var int: end_times = [start_times[i] + durations[i] | i in 1..num_tests];
array[1..num_resources] of var int: total_resource_usage = [sum(i in 1..num_tests where required_resources[i, r])(durations[i]) | r in 1..num_resources];
array[1..num_machines] of var int: machine_usage = [sum(i in 1..num_tests where assigned_machines[i] == m)(durations[i]) | m in 1..num_machines];

var int: makespan = max(end_times);

% Constraints

constraint makespan <= target_makespan;

constraint forall(i in 1..num_tests)(
    start_times[i] >= 0  % Ensure the test starts after time 0
);

constraint forall(i in 1..num_tests)(
    hardcoded_machines[i] == 0 \/
    assigned_machines[i] == hardcoded_machines[i]
);

constraint forall(i in 1..num_tests)(
    required_machines[i, assigned_machines[i]] == true
);




% Prevent overlapping of tests on the same machine
constraint forall(i, j in 1..num_tests where i > j)(
    (assigned_machines[i] == assigned_machines[j]) ->
    ((start_times[i] + durations[i] <= start_times[j]) \/
    (start_times[j] + durations[j] <= start_times[i]))
);


% Prevent overlapping of resource usage
constraint forall(r in 1..num_resources)(
    forall(i, j in 1..num_tests where i > j)(
        (required_resources[i, r] /\ required_resources[j, r]) ->
        ((start_times[i] + durations[i] <= start_times[j]) \/
        (start_times[j] + durations[j] <= start_times[i]))
    )
);


constraint forall(m in 1..num_machines)(
    exists(i in 1..num_tests where assigned_machines[i] == m)(
        start_times[i] == 0
    ) \/
    forall(i in 1..num_tests)(
        start_times[i] == 0 \/ not required_machines[i, m] \/
        exists(r in 1..num_resources where required_resources[i, r])(
            exists(j in 1..num_tests where j != i /\ required_resources[j, r])(
                start_times[j] == 0
            )
        )
    ) \/
    exists (i in 1..num_tests)(
        start_times[i] > 0 /\
        assigned_machines[i] == m /\
        exists(r in 1..num_resources where required_resources[i, r])(
            exists(j in 1..num_tests where j != i /\ required_resources[j, r])(
                start_times[j] == 0 /\ start_times[i] == durations[j]
            )
        )
    )
);

/*constraint forall(m in 1..num_machines)(
    forall (t1 in 1..num_tests where assigned_machines[t1] == m)(
        % either all tests are already assigned by the time t1 finishes
        forall (t2 in 1..num_tests where t2 > t1)(
            start_times[t2] <= start_times[t1] + durations[t1]
        ) \/
        % or the next test starts immediately after t1 finishes in the same machine
        exists(t2 in 1..num_tests where t1 > t2 /\ assigned_machines[t2] == m)(
            start_times[t2] == start_times[t1] + durations[t1]
        ) \/
        % or there are no tests that can be assigned to the machine
        forall(t2 in 1..num_tests where t2 > t1)(
            start_times[t2] < start_times[t1] + durations[t1] \/
            not required_machines[t2, m] \/
            exists(r in 1..num_resources where required_resources[t2, r])(
                exists(t3 in 1..num_tests where t3 > t2 /\ required_resources[t3,r])(
                    (start_times[t3] <= start_times[t2] /\ start_times[t2]<start_times[t3] + durations[t3]) \/
                    (start_times[t2] < start_times[t3] /\ start_times[t3] < start_times[t2] + durations[t2])
                )
            )
        ) \/
        % or there is a test t2 starting which uses a resource r that is being used by another test t3 already running when t1 finishes
        % and t2 is assigned to this machine when t3 finishes
        exists(t2 in 1..num_tests)(
            start_times[t2] > start_times[t1] + durations[t1] /\
            assigned_machines[t2] == m /\
            exists(r in 1..num_resources where required_resources[t2, r])(
                exists(t3 in 1..num_tests where t3 != t2 /\ required_resources[t3,r])(
                    start_times[t3] <= start_times[t2] /\ start_times[t2] < start_times[t3] + durations[t3] /\
                    start_times[t2] == start_times[t3] + durations[t3]
                )
            )
        )
    )
);*/
solve
%:: warm_start( start_times, initial_start_times)
%:: warm_start( assigned_machines, initial_assigned_machines)
%::int_search(assigned_machines, first_fail, indomain_min)

% do a priority search, using input_order
%::priority_search([tests_ordered[i] | i in 1..num_tests], assigned_machines, input_order, indomain_min)
satisfy;


