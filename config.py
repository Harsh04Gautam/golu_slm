class Config:
    # trian config
    total_iteration = 3001
    eval_iteration = 200
    eval_interval = 500
    generate_interval = 3000
    max_tokens = 1000
    save_model = False

    # model config
    batch = 4
    num_head = 64
    num_embed = 8 * num_head
    block = 1024
    stride = 4
    num_layer = 4
    dropout = 0.2
    learning_rate = 1e-3
