import pandas as pd
import numpy as np
import os
from PIL import Image
import yaml

if __name__ == "__main__":
    # Загрузка конфигурации из файла config.yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Чтение таблицы цветов из файла colors.xlsx
    colors_dir = config['data']['colors_dir']
    colors_df = pd.read_excel(colors_dir, engine='openpyxl')

    # Создание директорий для сохранения изображений
    train_dir = config['data']['train_dir']
    val_dir = config['data']['val_dir']
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Размер изображений
    img_size = config['data']['img_size']

    # Количество изображений на класс
    num_train_samples = config['data']['num_train_samples']
    num_val_samples = config['data']['num_val_samples']

    # Создание списка классов и маппинг цвет -> RGB
    classes = colors_df['Name'].tolist()
    class_to_rgb = {}
    for index, row in colors_df.iterrows():
        class_name = str(row['Name'])
        rgb_str = str(row['Rgb'])
        rgb = tuple(map(int, rgb_str.strip("()").split(",")))
        class_to_rgb[class_name] = rgb

    for class_name, rgb in class_to_rgb.items():
        # Создание директорий для каждого класса
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Генерация тренировочных изображений
        for i in range(num_train_samples):
            img_array = np.full((img_size, img_size, 3), rgb, dtype=np.uint8)
            img = Image.fromarray(img_array)

            img.save(os.path.join(train_class_dir, f"{class_name}_{i}.png"))

        # Генерация валидационных изображений
        for i in range(num_val_samples):
            img_array = np.full((img_size, img_size, 3), rgb, dtype=np.uint8)
            img = Image.fromarray(img_array)

            img.save(os.path.join(val_class_dir, f"{class_name}_{i}.png"))
