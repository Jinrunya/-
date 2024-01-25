import os

class config:
    # 根目录
    root_path = os.getcwd()
    data_dir = os.path.join(root_path, './data1/')
    train_data_path = os.path.join(root_path, 'train.json')
    output_path = os.path.join(root_path, 'output')
    output_test_path = os.path.join(output_path, 'test.txt')
    load_model_path = None


    # 一般超参
    epoch = 10
    learning_rate = 1e-5
    weight_decay = 0
    num_labels = 3
    loss_weight = [1,1,1]
    validation_split=0.1
    random_seed=42

    # Fuse相关
    fuse_model_type = 'NaiveCombine'
    only = None
    middle_hidden_size = 64
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.5
    out_hidden_size = 128

    # BERT相关
    fixed_text_model_params = False
    bert_name = 'roberta-base'
    bert_learning_rate = 1e-6
    bert_dropout = 0.2

    # ResNet相关
    fixed_img_model_params = False
    image_size = 224
    fixed_image_model_params = True
    resnet_learning_rate = 1e-6 
    resnet_dropout = 0.2
    img_hidden_seq = 64  


    # Dataloader params
    checkout_params = {'batch_size': 4, 'shuffle': False}
    train_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 2}
