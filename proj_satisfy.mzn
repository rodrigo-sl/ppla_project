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
int: num_problematic_resource_tests;
int: num_problematic_machine_tests;

array[1..num_tests] of int: durations;
array[1..num_tests, 1..num_machines] of bool: required_machines;
array[1..num_tests, 1..num_resources] of bool: required_resources;

%set of int: machines_ordered;  % Specific order of machines
%set of int: tests_ordered;  % Specific order of tests

array[1..num_eq_machines, 1..2] of int: eq_machines;
array[1..num_eq_tests, 1..2] of int: eq_tests;


% more problematic tests
%array[1..num_problematic_resource_tests] of int: problematic_resource_tests;
%array[1..num_problematic_machine_tests] of int: problematic_machine_tests;

array[1..num_tests_with_same_resources_splits + 1] of int: tests_with_same_resources_splits;
array[1..num_tests_with_same_resources] of int: tests_with_same_resources;

% Decision variables
array[1..num_tests] of var 1..num_machines: assigned_machines; % Machine assigned to each test
array[1..num_tests] of var 0..target_makespan - min_test_duration: start_times; % Start time for each test


array[1..num_resources + 1] of 1..total_resources_num + 1: resource_splits;
array[1..total_resources_num] of var int: tests_ordered_by_resource_ids;

% hardcoded arrays
array[1..num_tests] of 0..num_machines: hardcoded_machines;
% warm-start
%array[1..num_tests] of 1..num_machines: initial_assigned_machines;
%array[1..num_tests] of 0..target_makespan - min_test_duration: initial_start_times;

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

% This constraint forces each resource r to be used at least once by any test at timestep 0, unless the
% resource r is everytime used in a machine, where the time step 0 is already occupied by a test that uses a different resource r2
% or for all tests that use resource r, they also use another resource r2 being used at timestep 0
constraint forall(r in 1..num_resources)(
    % for each resource there is a test that starts at time 0
    exists (rs in resource_splits[r]..resource_splits[r+1] - 1)(
        start_times[tests_ordered_by_resource_ids[rs]] == 0
    ) \/
    % unless for all tests that use the same resource, they use at least another resource which is used by another test that starts at time 0
    forall(rs in resource_splits[r]..resource_splits[r+1] - 1)(
        exists(r2 in 1..num_resources where r2 != r /\ required_resources[tests_ordered_by_resource_ids[rs], r2])(
            exists(rs2 in resource_splits[r2]..resource_splits[r2+1] - 1)(
                start_times[tests_ordered_by_resource_ids[rs2]] == 0
            )
        )
    ) \/
    % or they are in a machine that has another test that starts at time 0 and uses another resource
    forall(rs in resource_splits[r]..resource_splits[r+1] - 1)(
        exists(t2 in 1..num_tests where t2 != tests_ordered_by_resource_ids[rs] /\ assigned_machines[t2] == assigned_machines[tests_ordered_by_resource_ids[rs]])(
            start_times[t2] == 0 /\
            exists(r2 in 1..num_resources where r2 != r /\ required_resources[t2, r2])(true)
        )
    )
);

% apply the same logic as the previous one but for the end
constraint forall(r in 1..num_resources)(
    resource_splits[r+1] - resource_splits[r] < 2 \/
    exists (rs in resource_splits[r]..resource_splits[r+1] - 1)(
        end_times[tests_ordered_by_resource_ids[rs]] == target_makespan
    ) \/
    forall(rs in resource_splits[r]..resource_splits[r+1] - 1)(
        exists(t in 1..num_tests_with_same_resources)(
            tests_with_same_resources[t] == tests_ordered_by_resource_ids[rs]
        )
    ) \/
    forall(rs in resource_splits[r]..resource_splits[r+1] - 1)(
        start_times[tests_ordered_by_resource_ids[rs]] == 0 \/
        exists(r2 in 1..num_resources where r2 != r /\ required_resources[tests_ordered_by_resource_ids[rs], r2])(
            exists(rs2 in resource_splits[r2]..resource_splits[r2+1] - 1)(
                end_times[tests_ordered_by_resource_ids[rs2]] == target_makespan
            )
        )
    ) \/
    forall(rs in resource_splits[r]..resource_splits[r+1] - 1)(
        start_times[tests_ordered_by_resource_ids[rs]] > 0 /\
        exists(t2 in 1..num_tests where t2 != tests_ordered_by_resource_ids[rs] /\ assigned_machines[t2] == assigned_machines[tests_ordered_by_resource_ids[rs]])(
            end_times[t2] == target_makespan /\
            exists(r2 in 1..num_resources where r2 != r /\ required_resources[t2, r2])(true)
        )
    )
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
constraint forall(t in 1..num_tests)(
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
);

% NOT WORKING
% forall tests t that have a set of resources s, if there is a test t2 which has a subset of s, then t2 should start after t
/*constraint forall(split in 1..num_resources)(
    forall(i,j in resource_splits[split]..resource_splits[split+1] - 1 where i!=j)(
        let {
            var int: t1 = tests_ordered_by_resource_ids[i];
            var int: t2 = tests_ordered_by_resource_ids[j];
        } in
        if (forall(r in 1..num_resources where required_resources[t2,r])(
            required_resources[t1,r]
        ) /\
        exists(r in 1..num_resources where required_resources[t1, r])(
            not required_resources[t2, r]
        )
        ) then
            start_times[t1] < start_times[t2]
        else
            true
        endif
    )
);*/

% for all tests that have EXACTLY the same set of resources, they must start one after the other
constraint forall(split in 1..num_tests_with_same_resources_splits - 1)(
    forall(i in tests_with_same_resources_splits[split]..tests_with_same_resources_splits[split + 1] - 1)(
        let {
            var int: t = tests_with_same_resources[i];
        } in
        start_times[t] == 0 \/
        end_times[t] == target_makespan \/
        exists(j in tests_with_same_resources_splits[split]..tests_with_same_resources_splits[split + 1] - 1 where j != i)(
            start_times[t] == end_times[tests_with_same_resources[j]] \/
            start_times[tests_with_same_resources[j]] == end_times[t]
        )
    )
);

% the objective of this this constraint is to try to reduce the search space by forcing tests that use the same resource
% to be used in the same machine as the tests with common resources that finish right before them.
constraint forall(r in 1..num_resources)(
    forall(i in resource_splits[r]..resource_splits[r+1] - 1)(
        let {
            var int: t1 = tests_ordered_by_resource_ids[i];
        } in
        start_times[t1] == 0 \/
        end_times[t1] == target_makespan \/
        exists(j in resource_splits[r]..resource_splits[r+1] - 1 where j != i)(
            let {
                var int: t2 = tests_ordered_by_resource_ids[j];
            } in
            (start_times[t1] == end_times[t2]) /\
            assigned_machines[t1] == assigned_machines[t2]
        ) \/
        exists( j in 1..num_resources where j != r /\ required_resources[t1, j])(
            exists(k in resource_splits[j]..resource_splits[j+1] - 1)(
                let {
                    var int: t2 = tests_ordered_by_resource_ids[k];
                } in
                (start_times[t1] == end_times[t2]) /\
                assigned_machines[t1] == assigned_machines[t2]
            )
        ) \/
        forall(j in 1..total_resources_num where tests_ordered_by_resource_ids[j] != t1)(
            let {
                var int: t2 = tests_ordered_by_resource_ids[j];
            } in
            end_times[t2] != start_times[t1] \/
            forall(k in 1..num_machines)(
                required_machines[t1, assigned_machines[t2]] == false
            )
        )
    )
);
% for all tests that have exactly the same resources (including tests that have no resources), if they start one after the other, 
% and in the same machine, then their position is fixed in relation to each other.
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
% tests with same resources must appear right after each other

% for all tests that need a resource r, they must start in the same machine of the previous test if possible


% if a test starts at a machine, it starts close to another test that uses the same machine




% a test that uses a resource must start

/*
constraint forall (split in 1..num_resources)(
    forall(i,j in resource_splits[split]..resource_splits[split + 1] - 1 where i!=j)(
        let {
            var int: t1 = tests_ordered_by_resource_ids[i];
            var int: t2 = tests_ordered_by_resource_ids[j];
        } in
        if (forall(r in 1..num_resources where required_resources[t2,r])(
            required_resources[t1,r]
        ) /\
        exists(r in 1..num_resources where required_resources[t1, r])(
            not required_resources[t2, r]
        )
        ) then
            let {
                var int: average_t1 = start_times[t1] + (durations[t1] div 2);
                var int: average_t2 = start_times[t2] + (durations[t2] div 2);
                var int: half_makespan = target_makespan div 2;
            } in
            (average_t1 > half_makespan /\ average_t2 < half_makespan) \/
            (average_t1 < half_makespan /\ average_t2 > half_makespan) \/
            (average_t1 > half_makespan /\ average_t2 > half_makespan /\ average_t1 > average_t2) \/
            (average_t1 < half_makespan /\ average_t2 < half_makespan /\ average_t1 < average_t2)
        else
            true
        endif
    )
);*/



%:: warm_start( start_times, initial_start_times)
%:: warm_start( assigned_machines, initial_assigned_machines)
%::int_search(start_times, most_constrained, indomain_min)

% do a priority search, using input_order
%::priority_search([tests_ordered[i] | i in 1..num_tests], assigned_machines, input_order, indomain_min)
%output[show(tests_ordered_by_resource)];  % Output the tests ordered by machine and start time
/*constraint forall(m in 1..num_machines)(
    (exists(i in 1..num_tests where assigned_machines[i] == m)(
        end_times[i] == target_makespan
    ) /\
    exists(i in 1..num_tests where assigned_machines[i] == m)(
        start_times[i] == 0
    )) \/
    forall(i in 1..num_tests)(
        not required_machines[i, m] \/
        (exists(r in 1..num_resources where required_resources[i, r])(
            exists(j in 1..num_tests where j != i /\ required_resources[j, r])(
                end_times[j] == target_makespan
            )
        ) /\
        exists(r in 1..num_resources where required_resources[i, r])(
            exists(j in 1..num_tests where j != i /\ required_resources[j, r])(
                start_times[j] == 0
            )
        ))

    ) \/
    (exists(i in 1..num_tests where assigned_machines[i] == m)(
        end_times[i] < target_makespan /\
        exists(r in 1..num_resources where required_resources[i, r])(
            exists(j in 1..num_tests where j != i /\ required_resources[j, r])(
                end_times[j] == target_makespan /\ end_times[i] == target_makespan - durations[j]
            )
        )
    ) /\
    exists(i in 1..num_tests where assigned_machines[i] == m)(
        start_times[i] > 0 /\
        exists(r in 1..num_resources where required_resources[i, r])(
            exists(j in 1..num_tests where j != i /\ required_resources[j, r])(
                start_times[j] == 0 /\ start_times[i] == durations[j]
            )
        )
    )
    )

);*/

/*constraint exists(i in 1..num_tests)(
    start_times[i] == 0 /\
    exists(r in 1..num_resources) (required_resources[i, r])
);
constraint exists(i in 1..num_tests)(
    end_times[i] == target_makespan /\
    exists(r in 1..num_resources) (required_resources[i, r])
);*/

/*constraint forall(i in 1..num_tests)(
    start_times[i] == 0 \/
    exists(j in 1..num_tests)(
        start_times[i] == end_times[j]
    )
);*/
% Prevent overlapping of resource usage
/*constraint forall(r in 1..num_resources)(
    forall(i in resource_splits[r]..resource_splits[r+1] - 2)(
        start_times[tests_ordered_by_resource[i]] + durations[tests_ordered_by_resource[i]] <= start_times[tests_ordered_by_resource[i+1]]
    )
);*/
/*constraint forall(m in 1..num_machines)(
    exists(i in 1..num_tests where assigned_machines[i] == m)(
        end_times[i] == target_makespan
    ) \/
    forall(i in 1..num_tests)(
        end_times[i] == target_makespan \/ not required_machines[i, m] \/
        exists(r in 1..num_resources where required_resources[i, r])(
            exists(j in 1..num_tests where j != i /\ required_resources[j, r])(
                start_times[j] + durations[j] < target_makespan - durations[i]
            )
        )
    ) \/
    exists (i in 1..num_tests)(
        end_times[i] < target_makespan /\
        assigned_machines[i] == m /\
        exists(r in 1..num_resources where required_resources[i, r])(
            exists(j in 1..num_tests where j != i /\ required_resources[j, r])(
                end_times[j] == target_makespan /\ end_times[i] == target_makespan - durations[j]
            )
        )
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
/*array[1..num_machines + 1] of var 1..num_tests + 1: machine_splits = 
    [ if m == 1 then 1 else 1 + count([assigned_machines[t] <= m-1 | t in 1..num_tests]) endif | m in 1..num_machines+1];
array[1..num_tests] of var 1..num_tests: tests_ordered_by_machine =
arg_sort([assigned_machines[t] * target_makespan + start_times[t] | t in 1..num_tests]);

array[1..total_resources_num] of int: tests_ordered_by_resource_ids;
array[1..total_resources_num] of var int: tests_ordered_by_resource_aux =
sort([ ((tests_ordered_by_resource_ids[id] div target_makespan) * target_makespan + start_times[tests_ordered_by_resource_ids[id] mod target_makespan]) * 1000 + tests_ordered_by_resource_ids[id] mod target_makespan | id in 1..total_resources_num]);
array[1..total_resources_num] of var 1..num_tests: tests_ordered_by_resource =
[tests_ordered_by_resource_aux[id] mod 1000 | id in 1..total_resources_num];*/

% Prevent overlapping of tests on the same machine
/*constraint forall (m in 1..num_machines)(
    forall(i in machine_splits[m]..machine_splits[m+1] - 2)(
        start_times[tests_ordered_by_machine[i]] + durations[tests_ordered_by_machine[i]] <= start_times[tests_ordered_by_machine[i+1]]
    )
);*/