include "cumulative.mzn";

int: num_tests;
int: num_resources;
int: min_makespan;
int: max_makespan;
int: total_resources_num;

array[1..num_tests, 1..num_resources] of bool: required_resources;
array[1..num_tests] of int: hardcoded_start_times;
array[1..num_tests] of int: durations;
array[1..num_resources + 1] of 1..total_resources_num + 1: resource_splits;
array[1..total_resources_num] of int: tests_ordered_by_resource_ids;

% decision variables
array[1..num_tests] of var 0 .. max_makespan - min(durations): start_times;

array[1..num_tests] of var int: end_times = [start_times[i] + durations[i] | i in 1..num_tests];

var min_makespan..max_makespan: makespan = max(end_times);

constraint forall(i in 1..num_tests)(
    if hardcoded_start_times[i] >= 0 then
        start_times[i] == hardcoded_start_times[i]
    else
        true
    endif
);

constraint forall (r in 1..num_resources)(
    cumulative(start_times, durations, [required_resources[i, r] | i in 1..num_tests], 1)
);

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
    )
);

constraint forall(t in 1..num_tests)(
    start_times[t] == 0 \/
    exists(r in 1..num_resources where required_resources[t, r])(
        exists(t2 in 1..num_tests where t2 != t /\ required_resources[t2, r])(
            start_times[t] == end_times[t2]
        )
    )
);


solve minimize makespan;