int: num_tests;
int: num_machines;
int: num_resources;
int: min_makespan;
int: max_makespan;
int: min_test_duration;

array[1..num_tests] of int: durations;
array[1..num_tests, 1..num_machines] of bool: required_machines;
array[1..num_tests, 1..num_resources] of bool: required_resources;

% Decision variables
array[1..num_tests] of var 1..num_machines: assigned_machines; % Machine assigned to each test
array[1..num_tests] of var 0..max_makespan - min_test_duration: start_times; % Start time for each test

% Compute end times for each test
array[1..num_tests] of var int: end_times = [start_times[i] + durations[i] | i in 1..num_tests];

var min_makespan..max_makespan: makespan = max(end_times);

% Constraints

constraint forall(i in 1..num_tests)(
    start_times[i] >= 0  % Ensure the test starts after time 0
);

constraint forall(i in 1..num_tests)(
    required_machines[i, assigned_machines[i]] == true
);

/*constraint forall(m, n in 1..num_machines where m > n)(
    abs(sum(i in 1..num_tests where assigned_machines[i] == m)(
        durations[i]
    ) - sum(j in 1..num_tests where assigned_machines[j] == n)(
        durations[j]
    )) <= max(t in 1..num_tests)(durations[t])
);*/

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

/*% whenever a test ends, there is another test that starts immediately after it, unless
% all tests are already assigned by the time the test finishes
constraint forall(i in 1..num_tests)(
    end_times[i] < max(end_times) ->
    exists(j in 1..num_tests where j != i)(
        start_times[j] == end_times[i]
    )
);*/


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
% Minimize makespan (the maximum end time)
solve minimize makespan;

/*constraint forall(i in 1..num_tests, j in 1..num_machines)(
    (assigned_machines[tests_ordered[i]] = machines_ordered[j]) -> 
    i <= j
);*/




