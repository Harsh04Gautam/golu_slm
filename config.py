class Config:
    # trian config
    epochs = 5
    save_model = True
    max_tokens = 500

    # model config
    batch = 16
    block = 512
    num_embed = 512
    head = num_embed
    num_head = 8
    kernel = 4
    num_layer = 8
    dropout = 0.0
    learning_rate = 8e-4

    def print_config(self):
        for name, value in self.__class__.__dict__.items():
            if name in {
                "epochs",
                "batch",
                "block",
                "num_head",
                "num_embed",
                "head",
                "kernel",
                "num_layer",
                "dropout",
                "learning_rate"
            }:
                print(f"{name:<20}: \033[1;92m{value}\033[0m")
        print("\n")
