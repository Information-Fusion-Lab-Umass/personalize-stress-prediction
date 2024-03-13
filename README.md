# Student Mental Health 

This is a repository for the paper titled "Dynamic clustering via branched deep learning enhances personalization of stress prediction from mobile sensor data" at Nature Scientific Report. 

## Relevant Resources

- [StudentLife study.](https://drive.google.com/open?id=171iN3Zis7SKJ-WqXzNmrhdEJut7aFqNS)
- [Extensive literature review on other psychology related papers for domain knowledge.](https://drive.google.com/open?id=1FyUFo0b3cYQv8KJJng-wvpgk_Uk53VI0)
- [Personalized Student Stress Prediction with Deep Multitask Network](https://arxiv.org/abs/1906.11356)

## Owners/Authors
- Yunfei Luo (yunfeiluo@umass.edu, yul268@ucsd.edu)
- Iman Deznaby (iman@cs.umass.edu)
- Abhinav Shaw (abhinav.shaw1993@gmail.com)
- Natcha Simsiri (natcha.simsiri@gmail.com)
- Dr. Tauhidur Rahman (trahman@ucsd.edu)
- Dr. Madalina Fiterau (mfiterau@cs.umass.edu)

## Input format
For each sample, the data needed are:
- ID of the subject that this sample belongs to (format: id_month_day_time)
- Time-series data (or features vector). 
- Covariate data (features vector). 
- Pre-processed data (e.g. histogram with 1 minute bins). 
- Label. 


For grouping, this have to be done before training or inference:
- Clustering the subjects based on chosen criteria (e.g. survey scores, DTW, or any relevant information). 
- Saved the result clusters as a dictionary in the format of "map: subject_id -> group_id". 
- If it is personalized model, then the group mapping should be one unique group_id for each subject_id. 


A sample fake data in sample_input.py indicate the desired format. 

## Training and Output
For training and validation:  
- Currently support 5-fold and leave-one-subject-out validation. 
- If custom train/val split is available, it can be added at the "get_splits" function in src/utils/train_val_utils.py. 
- the main training function is the "train_and_val" function in src/utils/train_val_utils.py. 
- The hyper-parameters are set under src/experiments/config.py. 

```
Training command:  
python3 -m src.experiments.run_exp job_session_name validation_type num_branches days_include cluster_name remark

Example:
Baseline: python3 -m src.experiments.repeat_exp lstm 5fold 1 0 all_in_one baseline
Calm-Net: python3 -m src.experiments.repeat_exp calm_net 5fold 1 0 one_for_each calm_net
Branching Calm-Net: python3 -m src.experiments.repeat_exp calm_net 5fold 3 0 one_for_each branching
Transformer example: python3 -m src.experiments.repeat_exp trans_calm_net_with_branching 5fold 1 0 one_for_each trans_calm_net_with_branching

LOOCV example with 1 week (7 days) included: python3 -m src.experiments.run_exp calm_net loocv 3 7 one_for_each branching
```

For experiments on WESAD:
- cd to folder wesad/
- Run command with example:
```
python3 -m main personalize loocv
```

For inference:
- The training function train_and_val function in src/utils/train_val_utils.py will return all the outputs including those from samples in validation set. 
- The output and evaluation scores are saved under data/cross_val_scores/  
- include keyword "test_only" in the job_session_name, and set the validation_type also to "test_only". 
