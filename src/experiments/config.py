
def get_config(job_name, device, num_features, student_groups, num_branches):
    config = {
        'model_params': {
            'device': device,
            'in_size': num_features,
            'AE': 'lstm', 
            'AE_num_layers': 1,
            'AE_hidden_size': 128,
            'shared_in_size': 128, # same with AE hidden
            'shared_hidden_size': 256,
            'num_branches': num_branches, # when equal to 1, it is equivalent to CALM_Net
            'groups': student_groups,
            'heads_hidden_size': 64,
            'num_classes': 3
            # 'num_classes': 2
        },
        'training_params': {
            'device': device, 
            'loss_weight': {
                'alpha': 1e-4,
                'beta': 1,
                'theta': 1 / 22, # 1 over number of students
            },
            'class_weights': [0.6456, 0.5635, 1.0000],
            # 'class_weights': [0.6456, 0.5635+1.0000],
            'global_lr': 1e-5,
            'branching_lr': 1e-5,
            'weight_decay': 1e-4,
            'epochs': 2, 
            'batch_size': 1,
            'use_histogram': True,
            'use_covariates': True, 
            'use_decoder': True,
        }
    }

    if job_name == 'calm_net':
        config['training_params']['global_lr'] = 1e-5 # 1e-6
        config['training_params']['branching_lr'] = 1e-5 # 1e-6
        config['training_params']['epochs'] = 200 # 500
    elif job_name == 'calm_net_with_branching':
        config['training_params']['global_lr'] = 1e-5
        config['training_params']['branching_lr'] = 1e-5
        config['training_params']['epochs'] = 200
    elif job_name in ['trans', 'trans_calm_net', 'trans_calm_net_with_branching']:
        config['model_params']['AE'] = 'trans'
        config['training_params']['global_lr'] = 1e-5
        config['training_params']['branching_lr'] = 1e-5
        config['training_params']['epochs'] = 100
        config['training_params']['use_decoder'] = False
    elif job_name == 'calm_net_no_cov':
        config['training_params']['global_lr'] = 1e-5
        config['training_params']['branching_lr'] = 1e-5
        config['training_params']['epochs'] = 200
        config['training_params']['use_covariates'] = False 
    elif job_name in ['lm_net', 'lstm']:
        config['training_params']['epochs'] = 200
        config['training_params']['use_decoder'] = False
    elif "test_only" in job_name:
        config["training_params"]["epochs"] = 1
    
    print('Num branches:', config['model_params']['num_branches'])
    return config
    