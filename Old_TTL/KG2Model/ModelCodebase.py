def model(name):
    code = ""
    if name == 'YOLO':
        code = """
from ultralytics import YOLO
model = YOLO("yolov8x.pt")
results = model(input_img)  # return a list of Results objects
result=results[0]
prediction = result.boxes # each boxes has the atribut:- xyxy (coodinates) -conf (confidence) -cls (class id in COCO format)
"""
        return code, ["COCO", "BoundingBox", "YOLO", "x1,y2,x2,y2,conf,class"]
    elif name == 'SAM':
        code = """
from ultralytics import SAM
model = SAM('sam_b.pt')
# Display model information (optional)
# model.info()
# Run inference with bboxes prompt
prediction = model()[0]
"""
        output_describtion = """#prediction output descrbtion:
    prediction.orig_img (numpy.ndarray): original image
    prediction.names (dict): dictionary of class names
    prediction.boxes (torch.tensor): 2D tensor of bounding box coordinates.
    prediction.masks.data.cpu() (torch.tensor): 3D tensor of detection masks, where each mask is a binary  (boolean).
    prediction.probs (torch.tensor): 1D tensor of probabilities of each class .
    prediction.keypoints (List[List[float]], optional): list of detected keypoints for each object.
boxes Attributes:
    boxes.xyxy (torch.Tensor | numpy.ndarray): boxes in xyxy format
    boxes.conf (torch.Tensor | numpy.ndarray):  confidence values
    boxes.cls (torch.Tensor | numpy.ndarray): class values
    boxes.id (torch.Tensor | numpy.ndarray): track IDs of the boxes (if available).
"""
        return code, ["SAM", output_describtion]
    elif name == 'OneFormer':
        code = """
from transformers import pipeline
classifier = pipeline('object-detection',model="shi-labs/oneformer_coco_swin_large")
from PIL import Image
prediction= classifier(Image.fromarray(input_img))
"""
        return code, ["BoundingBox", "OneFormer"]
    elif name in ['YOLOS','DETR','DETA']:
        pred_out = "prediction_output={'box': {'xmax': , 'xmin': , 'ymax': , 'ymin': 452}, 'label': , 'score': }"
        model_dict = {'YOLOS':"\"hustvl/yolos-tiny\"",
                      'DETR':"\"facebook/detr-resnet-50\"",
                      'DETA':"\"jozhang97/deta-resnet-50\""}
        code = """
from PIL import Image
from transformers import pipeline
classifier = pipeline('object-detection',model="""+model_dict[name]+""")
prediction= classifier(Image.fromarray(input_img))
"""
        return code, ["BoundingBox", name,pred_out]
    elif name == 'BERT':
        code = """
def predict_sentiment(text):
    #returns list of dictinary with a 'label' that rates the sentiment from 1-5 starts i.e. '5 stars' and a 'score' (flaot)
    from transformers import pipeline
    classifier = pipeline('text-classification', model="nlptown/bert-base-multilingual-uncased-sentiment")
    prediction = classifier(text)
    return prediction

prediction=predict_sentiment(text)
"""
        return code, ["BERT", "Sentiment Analysis"]
    elif name == 'distilbert':
        code = """
def predict_sentiment(text):
    #returns list of alist of dictinarys one for each label. they include the keys 'label' ('positive', 'negative', 'neutral') and a 'score' (flaot)
    from transformers import pipeline
    classifier = pipeline('text-classification', model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    prediction = classifier(text)
    return prediction

prediction=predict_sentiment(text)
"""
        return code, ["distilbert", "Sentiment Analysis"]
    elif name == 'VIT':
        code = """
from PIL import Image
from transformers import pipeline
classifier = pipeline('image-classification', model="google/vit-base-patch16-224")
#returns list of dictinary with a 'label' (class) and a 'score' (flaot)
prediction = classifier(Image.fromarray(input_img))
"""
        return code, ["distilbert", "Sentiment Analysis"]
    elif name == 'Inceptionv4':
        code = """
def predict_img_class(input_image):
    # Imports
    import timm
    import torch
    import numpy as np
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from RDF_GPT_TOOL.utils import getImagenet_1k_idx_to_class


    # Load model
    model = timm.create_model('inception_v4', pretrained=True)
    model.eval()

    # Prepare the transformation
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    # Transform the input image
    tensor = transform(Image.fromarray(input_img)).unsqueeze(0)

    # Predict with the model
    with torch.no_grad():
        out = model(tensor)

    # Process the output
    pred = torch.nn.functional.softmax(out[0], dim=0)
    id_to_class = getImagenet_1k_idx_to_class()
    prediction_idx = np.argpartition(pred, -5)[-5:]

    # Generate output
    output = {id_to_class[i.item()]: pred[i].item() for i in prediction_idx}
    return output

prediction = predict_img_class(input_img)
"""
        return code, ["Inceptionv4", "Sentiment Analysis"]

    return code, []
