import os, sys
sys.path.append("/home/summer_intern/Seongsu/crp_pytorch/pytorch-grad-cam")
from torchvision import models
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReLU
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
from CNN_utils import *

class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)

class DifferenceFromConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return 1 - cos(model_output, self.features)

def restore_image(image):
    mean = pt.tensor([0.485, 0.456, 0.406])
    std = pt.tensor([0.229, 0.224, 0.225])
    image = image.permute(1,2,0)
    image = std * image + mean
    return image

grad_dict = {"GradCAM":GradCAM,
             "GradCAMPlusPlus":GradCAMPlusPlus,
             "EigenGradCAM":EigenGradCAM,
             "AblationCAM":AblationCAM,
             "RandomCAM":RandomCAM
            }

class ClassifierOutputReLU_T:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
#             return torch.nn.ReLU()(torch.tanh(model_output))[self.category]
            return torch.nn.ReLU()(torch.nn.Tanh()(model_output))[self.category]
#         return torch.nn.ReLU()(torch.tanh(model_output))[:, self.category]
        return torch.nn.ReLU()(torch.nn.Tanh()(model_output))[:, self.category]

# class ClassifierOutputReLU_T2:
#     def __init__(self, category):
#         self.category = category

#     def __call__(self, model_output):
#         if len(model_output.shape) == 1:
# #             return torch.nn.ReLU()(torch.tanh(model_output))[self.category]
#             return torch.nn.ReLU()(torch.heaviside(model_output,torch.tensor(0)))[self.category]
# #         return torch.nn.ReLU()(torch.tanh(model_output))[:, self.category]
#         return torch.nn.ReLU()(torch.heaviside(model_output,torch.tensor(0)))[:, self.category]

def build_grad(model, samples, concept_ids, concept_dict, target_layer, x_list, y_list, pred_list, hidden_sample_list, classes, grad_func = "AblationCAM"):
    dict = {c_id:[] for c_id in concept_ids}
    target_layers = [target_layer]
    for c_id in concept_ids:
        for data_index in samples:
            img, y = x_list[data_index], int(torch.argmax(y_list[data_index]).item())
            input_tensor = img.unsqueeze(0)
            img = np.array(restore_image(img))
            
            targets = [ClassifierOutputTarget(c_id)]
#             targets = [ClassifierOutputReLU(c_id)]
            func = GradCAM
            if grad_func in grad_dict:
                func = grad_dict[grad_func]
            with func(model=model, target_layers=target_layers, use_cuda = True) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
            dict[c_id].append(cam_image)
    return [dict, samples, concept_ids, concept_dict]


def build_grad_pn(model, samples, concept_ids, concept_dict, target_layer, x_list, y_list, pred_list, hidden_sample_list, classes, grad_func = "AblationCAM"):
    dict = {c_id:[] for c_id in concept_ids}
    target_layers = [target_layer]
    for c_id in concept_ids:
        for data_index in samples:
            img, y = x_list[data_index], int(torch.argmax(y_list[data_index]).item())
            input_tensor = img.unsqueeze(0)
            img = np.array(restore_image(img))
            data = pt.zeros(hidden_sample_list.shape[1])
            data[c_id] = 1
            
            if concept_dict[c_id]:
                targets = [SimilarityToConceptTarget(data.cuda())]
            else :
                targets = [DifferenceFromConceptTarget(data.cuda())]

            func = GradCAM
            if grad_func in grad_dict:
                func = grad_dict[grad_func]
            with func(model=model, target_layers=target_layers, use_cuda = True) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
            dict[c_id].append(cam_image)
    return [dict, samples, concept_ids, concept_dict]

def build_grad_sum(model, samples, concept_ids, concept_dict, target_layer, x_list, y_list, pred_list, hidden_sample_list, classes, grad_func = "AblationCAM"):
    dict = {c_id:[] for c_id in concept_ids}
    dict_gray = {c_id:[] for c_id in concept_ids}

    target_layers = [target_layer]
    for c_id in concept_ids:
        for data_index in samples:
            img, y = x_list[data_index], int(torch.argmax(y_list[data_index]).item())
            input_tensor = img.unsqueeze(0)
            img = np.array(restore_image(img))
            
            data = pt.zeros(256)
#             data = copy.deepcopy(hidden_sample_mean)
#             data = hidden_sample_list[data_index]

            data[c_id] = hidden_sample_list[data_index][c_id]
#             targets = [DifferenceFromConceptTarget(data.cuda())]
            targets = [ClassifierOutputTarget(c_id)]
#             targets = [Nothing()]

            func = GradCAM
            if grad_func in grad_dict:
                func = grad_dict[grad_func]
            with func(model=model, target_layers=target_layers, use_cuda = True) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                dict_gray[c_id].append(grayscale_cams)
                cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
            dict[c_id].append(cam_image)
            
    dict_sum = {idx:np.zeros_like(grayscale_cams) for idx in samples}

    for i, data_index in enumerate(samples):
        for c_id in concept_ids:
            if concept_dict[c_id]:
                dict_sum[data_index] += dict_gray[c_id][i]/len(concept_ids)
            else:
                dict_sum[data_index] -= dict_gray[c_id][i]/len(concept_ids)
        img = np.array(restore_image(x_list[data_index]))
        dict_sum[data_index] = show_cam_on_image_2(img, NormalizeData(dict_sum[data_index][0, :]), use_rgb=True)
    return [dict, samples, concept_ids, concept_dict] , dict_sum


def grad_plot(dict, samples, concept_ids, concept_dict, x_list, y_list, pred_list, hidden_sample_list, classes):
    # === Plot Image and LRP ===
    fig, axes = plt.subplots(1,len(samples), figsize=(20,2), facecolor="white")
    for i, index in enumerate(samples):
        img = x_list[index]
        img = restore_image(img)
        axes[i].imshow(img)
        axes[i].set_title("{}".format(classes[int(torch.argmax(y_list[index]).item())]))
        axes[0].set_ylabel("image", fontsize=20)

    # === Plot CRP ===
    fig, axes = plt.subplots(len(concept_ids), len(samples), figsize=(20,len(concept_ids)*2), facecolor="white")
    keys = list(dict.keys())
    if len(keys) == 1:
        for i, key in enumerate(keys):
            for j, item in enumerate(dict[key]):
#             item = item.transpose(1,2,0)
                axes[j].imshow(item)
                axes[0].set_ylabel(str(key)+" "+ str(concept_dict[key]), fontsize=20)
    else:
        for i, key in enumerate(keys):
            for j, item in enumerate(dict[key]):
    #             item = item.transpose(1,2,0)
                axes[i,j].imshow(item)
                axes[i,0].set_ylabel(str(key)+" "+ str(concept_dict[key]), fontsize=20)
    plt.tight_layout()
    
def grad_plot_sum(dict, samples, concept_ids, concept_dict, dict_sum, x_list, y_list, pred_list, hidden_sample_list, classes):
    # === Plot Image and LRP ===
    fig, axes = plt.subplots(1,len(samples), figsize=(20,2), facecolor="white")
    for i, index in enumerate(samples):
        img = x_list[index]
        img = restore_image(img)
        axes[i].imshow(img)
        axes[i].set_title("{}".format(classes[int(torch.argmax(y_list[index]).item())]))
        axes[0].set_ylabel("image", fontsize=20)

    # === Plot CRP ===
    fig, axes = plt.subplots(len(concept_ids)+1, len(samples), figsize=(20,len(concept_ids)*2+2), facecolor="white")
    keys = list(dict.keys())
    for i, key in enumerate(keys):
        for j, item in enumerate(dict[key]):
            axes[i,j].imshow(item)
            axes[i,0].set_ylabel(str(key)+" "+ str(concept_dict[key]), fontsize=20)
    for j, key in enumerate(samples):
        axes[-1,j].imshow(dict_sum[key])        
    plt.tight_layout()

    
def show_cam_on_image_2(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
#     if use_rgb:
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cm = plt.get_cmap('seismic')
#     print(mask.shape)
    heatmap = cm(mask)[:,:,:3]
#     print(heatmap)
#     print(heatmap.shape)

#     heatmap = np.float32(heatmap)

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def NormalizeData(data):
    data[data < 0] /= abs(np.min(data))
    data[data > 0] /= abs(np.max(data))
    return data