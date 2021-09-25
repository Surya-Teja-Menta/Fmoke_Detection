from PIL import Image
from torchvision.transforms import transforms
import torch


class_names = class_names = ['Fire', 'Neutral', 'Smoke']

def Predict(image):
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    model = torch.load('saved_models/model_final.pth',map_location=torch.device('cpu'))
    pred = model(image)
    idx = torch.argmax(pred)
    prob = pred[0][idx].item()*100
    
    return class_names[idx], prob

def predict(path):
    img = Image.open(path)
    prediction, prob = predict(img)
    print(prediction, prob)
    return prediction, prob
    


if __name__=='__main__':
    predict(path=None)
    Predict(path=None)