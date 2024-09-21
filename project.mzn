% input data

int: n_tests;
int: n_machines;
int: n_resources;

array[1..n_tests] of int: durations;

array[1..n_tests, 1..n_machines] of 0..1: possible_machines;
array[1..n_tests, 1..n_resources] of 0..1: resource_requirements;

% output data

var int: makespan;

array[1..n_tests] of var 1..n_machines: test_machine_assignment;
array[1..n_tests] of var 0..makespan: test_start_times;  