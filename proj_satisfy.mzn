% Use this editor as a MiniZinc scratch book
include "cumulative.mzn";
int: num_tests;
int: num_machines;
int: num_resources;
int: target_makespan;
int: min_test_duration;
int: total_resources_num;
int:num_eq_machines;
int:num_eq_tests;
int:num_tests_with_same_resources;
int:num_tests_with_same_resources_splits;

array[1..num_tests] of int: durations;
array[1..num_tests, 1..num_machines] of bool: required_machines;
array[1..num_tests, 1..num_resources] of bool: required_resources;

array[1..num_eq_machines, 1..2] of int: eq_machines;
array[1..num_eq_tests, 1..2] of int: eq_tests;


array[1..num_tests_with_same_resources_splits + 1] of int: tests_with_same_resources_splits;
array[1..num_tests_with_same_resources] of int: tests_with_same_resources;

array[1..num_tests] of int: resource_start_times;
% Decision variables
array[1..num_tests] of var 1..num_machines: assigned_machines; % Machine assigned to each test
array[1..num_tests] of var 0..target_makespan - min_test_duration: start_times; % Start time for each test


array[1..num_resources + 1] of 1..total_resources_num + 1: resource_splits;
array[1..total_resources_num] of var int: tests_ordered_by_resource_ids;

% hardcoded arrays
array[1..num_tests] of 0..num_machines: hardcoded_machines;


% Compute end times for each test
array[1..num_tests] of var int: end_times = [start_times[i] + durations[i] | i in 1..num_tests];

var int: makespan = max(end_times);

% Constraints

constraint makespan <= target_makespan;

% break symmetry between parallel machines, by ensuring that the first test of every machine has a lower index than the first test of the parallel machine
constraint forall(i in 1..num_tests)(
    start_times[i] >= 0  % Ensure the test starts after time 0
);

constraint forall(i in 1..num_tests)(
    if resource_start_times[i] >= 0 then
        start_times[i] == resource_start_times[i]
    else
        true
    endif
);

constraint forall(i in 1..num_tests)(
    hardcoded_machines[i] == 0 \/
    assigned_machines[i] == hardcoded_machines[i]
);


constraint forall(i in 1..num_tests)(
    required_machines[i, assigned_machines[i]] == true
);


constraint forall (r in 1..num_resources)(
    cumulative(start_times, durations, [required_resources[i, r] | i in 1..num_tests], 1)
);

constraint forall(i in 1..num_machines)(
    cumulative(start_times, durations, [if assigned_machines[j] == i then 1 else 0 endif | j in 1..num_tests], 1)
);

constraint forall(eq_pair in 1..num_eq_tests)(
    assigned_machines[eq_tests[eq_pair, 1]] <= assigned_machines[eq_tests[eq_pair, 2]]
);

constraint forall(eq_pair in 1..num_eq_machines)(
    [i | i in 1..num_tests where assigned_machines[i] == eq_machines[eq_pair, 1]][1] <= [i | i in 1..num_tests where assigned_machines[i] == eq_machines[eq_pair, 2]][1]
);


% forces all machines to have a test running at timestep 0, unless there is no test available to run at that
% time in that machine or if in that machine, there is scheduled a test that needs a resource already used
% by another test starting at timestep 0
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

% for a test to start in a machine, it must start close to another test that uses the same machine
% the idea is to prevent one test from being scheduled in the middle of nowhere in a machine, forcing
% it to be close to another test.
/*constraint forall(t in 1..num_tests)(
    start_times[t] == 0 \/
    end_times[t] == target_makespan \/
    % if its a test without resources must always start close to another test
    exists(t2 in 1..num_tests where t2 != t /\ assigned_machines[t2] == assigned_machines[t])(
        start_times[t] == end_times[t2] \/
        start_times[t2] == end_times[t]
    ) \/
    % its a test with resources
    exists(r in 1..num_resources where required_resources[t, r])(
        exists(i in resource_splits[r]..resource_splits[r+1]-1)(
            (start_times[tests_ordered_by_resource_ids[i]] == end_times[t] \/
            start_times[t] == end_times[tests_ordered_by_resource_ids[i]])
        )
    )
);*/
/*
% tests that use exactly the same resources must appear one after the other
constraint forall(split in 1..num_tests_with_same_resources_splits - 1)(
    forall(i in tests_with_same_resources_splits[split]..tests_with_same_resources_splits[split + 1] - 1)(
        let {
            var int: t = tests_with_same_resources[i];
        } in
        start_times[t] == 0 \/
        end_times[t] == target_makespan \/
        forall(j in tests_with_same_resources_splits[split]..tests_with_same_resources_splits[split + 1] - 1 where j != i)(
            start_times[t] < start_times[tests_with_same_resources[j]]
        ) \/
        exists(j in tests_with_same_resources_splits[split]..tests_with_same_resources_splits[split + 1] - 1 where j != i)(
            start_times[t] == end_times[tests_with_same_resources[j]] \/
            start_times[tests_with_same_resources[j]] == end_times[t]
        )
    )
);
*/
% tests that use exactly the same set of resources, including tests which don't use any resources at all,
% if running on the same machine, must be ordered by start time
constraint forall(split in 1..num_tests_with_same_resources_splits)(
    forall(i in tests_with_same_resources_splits[split]..tests_with_same_resources_splits[split + 1] - 2,
    j in i+1..tests_with_same_resources_splits[split] - 1)(
        let {
            var int: t1 = tests_with_same_resources[i];
            var int: t2 = tests_with_same_resources[j];
        } in
        if assigned_machines[t1] == assigned_machines[t2] /\
        not exists(l in 1..num_tests where l != t1 /\ l != t2 /\ assigned_machines[l] == assigned_machines[t1])(
            (start_times[l] > t1 /\ start_times[l] < t2) \/
            (start_times[l] < t1 /\ start_times[l] > t2)
        ) then
            start_times[t1] < start_times[t2]
        endif
    )
);

solve satisfy;