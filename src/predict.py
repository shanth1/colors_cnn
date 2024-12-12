import torch
from torchvision import transforms
from PIL import Image
import yaml
from model_train import ColorCNN
from model_train import ColorDataset

def create_transform(input_size):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])

def predict_image(image_path, model, transform, device):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return pred.item()

def index_to_class(class_to_idx):
    return {v: k for k, v in class_to_idx.items()}

# ========================================

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    use_gpu = False
    if config['model']['use_gpu']:
        use_gpu = torch.cuda.is_available()

    device = torch.device("cuda" if use_gpu else "cpu")
    transform = create_transform(config['model']['input_size'])

    dataset = ColorDataset(config['data']['train_dir'], transform)
    class_to_idx = dataset.class_to_idx
    idx_to_class = index_to_class(class_to_idx)

    num_classes = len(class_to_idx)

    model = ColorCNN(config, num_classes)
    model.load_state_dict(torch.load(config["predict"]["model_path"], map_location=device))
    model = model.to(device)
    model.eval()

    image_path = config['predict']['image_path']
    predicted_class_idx = predict_image(image_path, model, transform, device)
    predicted_class = idx_to_class[int(predicted_class_idx)]
    print(f'Prediction: {predicted_class}')
