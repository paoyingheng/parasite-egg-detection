import gradio as gr
import torch
from ultralyticsplus import YOLO, render_result

torch.hub.download_url_to_file('https://as1.ftcdn.net/v2/jpg/01/85/59/30/1000_F_185593012_ed2xkZFSC9B66fNCBkoURPYht8dwRjJw.jpg', 'one.jpg')
torch.hub.download_url_to_file('https://st4.depositphotos.com/3687893/27930/i/450/depositphotos_279301742-stock-photo-parasite-egg-ascaris-lumbricoides-find.jpg', 'two.jpg')
torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Trichuris_trichiura_egg_%2801%29.tif/lossy-page1-1200px-Trichuris_trichiura_egg_%2801%29.tif.jpg', 'three.jpg')

def para_func(image: gr.Image = None, image_size: gr.Slider = 640, conf_threshold: gr.Slider = 0.4, iou_threshold: gr.Slider = 0.50):
    model = YOLO('best.pt')

    # Perform object detection on the input image using the YOLO model
    results = model.predict(image, conf=conf_threshold, iou=iou_threshold, imgsz=image_size)

    # Print the detected objects' information (class, coordinates, and probability)
    box = results[0].boxes
    print("Object type:", box.cls)
    print("Coordinates:", box.xyxy)
    print("Probability:", box.conf)

    # Render the output image with bounding boxes around detected objects
    render = render_result(model=model, image=image, result=results[0])
    return render

# Define input and output components for Gradio interface
inputs = [
    gr.Image(type="filepath", label="Input Image"),
    gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="Image Size"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold"),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.45, step=0.05, label="IOU Threshold"),
]

outputs = gr.Image(type="filepath", label="Output Image")

title = "Parasitic Egg Detection and Classification in Microscopic Images with YOLOv8"

examples = [['one.jpg', 640, 0.5, 0.7],
            ['two.jpg', 800, 0.5, 0.6],
            ['three.jpg', 900, 0.5, 0.8]]

# Creating the Gradio interface
yolo_app = gr.Interface(
    fn=para_func,
    inputs=inputs,
    outputs=outputs,
    title=title,
    examples=examples,
    cache_examples=True,
)

# Launch the Gradio interface in debug mode with queue enabled
yolo_app.launch(debug=True)
