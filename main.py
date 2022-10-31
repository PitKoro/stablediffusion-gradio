from tkinter.tix import InputOnly
import numpy as np
import gradio as gr

from model import *
from utils import *


device = 'cuda'

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
model = model.load_from_checkpoint('checkpoints/epoch=285-step=10000.ckpt', lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
model.cuda()
model.eval()




def sepia(input_img):
    print(f"type(input_img) = {type(input_img)}")
    input_img = Image.fromarray(input_img)
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    print(f"type(sepia_img) = {type(sepia_img)}")
    return sepia_img



def predict(image):
    image = Image.fromarray(image)
    convert_tensor = transforms.ToTensor()
    tensor_image = convert_tensor(image).unsqueeze_(0).to(device)
    outputs = model(pixel_values=tensor_image, pixel_mask=None)
    return np.array(visualize_predictions(image, outputs))




def main():
    demo = gr.Interface(predict, gr.Image(), "image")
    demo.launch(server_name="0.0.0.0", server_port=7860)
    
if __name__ == "__main__":
    main()