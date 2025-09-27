class Config:
    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "efficientnet_b4"
    RANDOM_STATE = 42
    TARGET_MAE = 50
    
    # Какие слои размораживаем - совпадают с неймингом в моделях
    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"  
    IMAGE_MODEL_UNFREEZE = "layer.3|layer.4"  
    
    # Гиперпараметры
    BATCH_SIZE = 32
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    REGRESSOR_LR = 5e-3
    EPOCHS = 100
    DROPOUT = 0.3
    HIDDEN_DIM = 256
    
    # Пути
    SAVE_PATH = "best_model.pth"
    COLAB = False
    ROOT_PATH = "/content/drive/MyDrive/ydsc/" if COLAB else "./"
    IMAGES_PATH = 'data/images'

