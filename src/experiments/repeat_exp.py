from src.experiments.run_exp import *

if __name__ == '__main__':
    repeat_n = 10

    # read command line arguments
    job_name = sys.argv[1] # choice: 'test', 'calm_net', 'calm_net_with_branching'
    split_name = sys.argv[2] # choice: ['5fold', 'loocv', '5fold_c']
    num_branches = int(sys.argv[3]) # any interger
    days_include = int(sys.argv[4]) # any interger
    clusters_name = sys.argv[5] # 'one_for_each', 'all_in_one', 'pre_survey_scores_7
    remark = sys.argv[6] # calm_net, single, survey

    for i in range(repeat_n):
        curr_remark = remark + str(i)
        run_exp(job_name, split_name, num_branches, days_include, clusters_name, curr_remark)