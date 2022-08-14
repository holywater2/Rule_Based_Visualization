from src.lrp_for_vgg import construct_lrp
from src.lrp import process_lrp_before_imshow
from CNN_utils import *

def compute_lrp(lrp_model, x, y = None, class_specific = True):
	# computation of lrp 
	x = x.unsqueeze(0)
	output = lrp_model.forward(x, y=y, class_specific=class_specific)
	all_relevnace = output['all_relevnaces']
	return all_relevnace

def compute_negative_lrp(lrp_model, x, y = None, class_specific = True):
	# computation of lrp 
	x = x.unsqueeze(0)
	output = lrp_model.neg_forward(x, y=y, class_specific=class_specific)
	all_relevnace = output['all_relevnaces']
	return all_relevnace

def restore_image(image):
    mean = pt.tensor([0.485, 0.456, 0.406])
    std = pt.tensor([0.229, 0.224, 0.225])
    image = image.permute(1,2,0)
    image = std * image + mean
    return image

def plot_hidden_LRP(dict, samples, concept_ids, concept_dict, lrp_list, hidden_sample_mean, x_list, y_list, pred_list, hidden_sample_list, classes):
    # === Plot Image and LRP ===
    fig, axes = plt.subplots(2, len(samples), figsize=(20,4), facecolor="white")
    for i, index in enumerate(samples):
    #     img = testset[index][0]
    #     img = load_image(index)[0]
        img = x_list[index]
        img = restore_image(img)
        axes[0,i].imshow(img)
    #     print(index,int(torch.max(y_list[index],dim=0).item()),y_list[index])
        axes[0,i].set_title("{}".format(classes[int(torch.argmax(y_list[index]).item())]))
    for j in range(len(samples)):
        item = lrp_list[j]
        item = item.permute(1,2,0).cpu().detach().numpy()
        R = item.sum(axis=-1)
        R, kwargs = process_lrp_before_imshow(R)
        axes[1,j].imshow(R, **kwargs)

    axes[0,0].set_ylabel("Image", fontsize=20)
    axes[1,0].set_ylabel("LRP",  fontsize=20)
    plt.tight_layout()

    # === Plot CRP ===
    fig, axes = plt.subplots(len(concept_ids)+1, len(samples), figsize=(20,len(concept_ids)*2+2), facecolor="white")
    keys = list(dict.keys())
    Rsum = [np.zeros_like(R) for i in range(len(samples))]
    for i, key in enumerate(keys):
        for j, item in enumerate(dict[key]):
            item = item.permute(1,2,0).cpu().detach().numpy()
            R = item.sum(axis=-1)
            R, kwargs = process_lrp_before_imshow(R)
            axes[i,j].imshow(R, **kwargs)
    #         print(kwargs)
            axes[i,0].set_ylabel(str(key)+" "+ str(concept_dict[key]), fontsize=20)
            Rsum[j] += R * (hidden_sample_list[samples[i]][key]- hidden_sample_mean[key]).item()
    #         if concept_dict[key]:
    #             Rsum[j] += R * (hidden_sample_list[samples[i]][key] - hidden_sample_mean[key]).item()/(hidden_sample_list[samples[i]][key] - hidden_sample_mean[key]).item()
    #         else :
    #             Rsum[j] += R * (hidden_sample_list[samples[i]][key] - hidden_sample_mean[key]).item()/hidden_sample_mean[key].item()
    #         Rsum[j] += R
    for j, Rs in enumerate(Rsum):
        Rs, kwargs = process_lrp_before_imshow(Rs)
        axes[-1,j].imshow(Rs,**kwargs)
    plt.tight_layout()

def plot_CRP(dict, samples, concept_ids, concept_dict, lrp_list, hidden_sample_mean, x_list, y_list, pred_list, hidden_sample_list, classes):
        # === Plot Image and LRP ===
    fig, axes = plt.subplots(2, len(samples), figsize=(20,4), facecolor="white")
    for i, index in enumerate(samples):
    #     img = testset[index][0]
    #     img = load_image(index)[0]
        img = x_list[index]
        img = restore_image(img)
        axes[0,i].imshow(img)
        axes[0,i].set_title("{}".format(classes[int(torch.argmax(y_list[index]).item())]))
    for j in range(len(samples)):
        item = lrp_list[j]
        item = item.permute(1,2,0).cpu().detach().numpy()
        R = item.sum(axis=-1)
        R, kwargs = process_lrp_before_imshow(R)
        axes[1,j].imshow(R, **kwargs)

    axes[0,0].set_ylabel("Image", fontsize=20)
    axes[1,0].set_ylabel("LRP",  fontsize=20)
    plt.tight_layout()

    # === Plot CRP ===
    fig, axes = plt.subplots(len(concept_ids), len(samples), figsize=(20,len(concept_ids)*2), facecolor="white")
    keys = list(dict.keys())
    if(len(concept_ids) == 1):
        for i, key in enumerate(keys):
            for j, item in enumerate(dict[key]):
                item = item.permute(1,2,0).cpu().detach().numpy()
                R = item.sum(axis=-1)
                R, kwargs = process_lrp_before_imshow(R)
                axes[j].imshow(R, **kwargs)
                axes[0].set_ylabel(str(key)+" "+str(concept_dict[key]), fontsize=20)
    else:
        for i, key in enumerate(keys):
            for j, item in enumerate(dict[key]):
                item = item.permute(1,2,0).cpu().detach().numpy()
                R = item.sum(axis=-1)
                R, kwargs = process_lrp_before_imshow(R)
                axes[i,j].imshow(R, **kwargs)
                axes[i,0].set_ylabel(str(key)+" "+str(concept_dict[key]), fontsize=20)
    plt.tight_layout()

    
### Old co
# def plot_hidden_LRP(dict, samples):
#     # === Plot Image and LRP ===
#     fig, axes = plt.subplots(2, len(samples), figsize=(20,4), facecolor="white")
#     for i, index in enumerate(samples):
#     #     img = testset[index][0]
#     #     img = load_image(index)[0]
#         img = x_list[index]
#         img = restore_image(img)
#         axes[0,i].imshow(img)
#     #     print(index,int(torch.max(y_list[index],dim=0).item()),y_list[index])
#         axes[0,i].set_title("{}".format(classes[int(torch.argmax(y_list[index]).item())]))
#     for j in range(len(samples)):
#         item = lrp_list[j]
#         item = item.permute(1,2,0).cpu().detach().numpy()
#         R = item.sum(axis=-1)
#         R, kwargs = process_lrp_before_imshow(R)
#         axes[1,j].imshow(R, **kwargs)

#     axes[0,0].set_ylabel("Image", fontsize=20)
#     axes[1,0].set_ylabel("LRP",  fontsize=20)
#     plt.tight_layout()

#     # === Plot CRP ===
#     fig, axes = plt.subplots(len(concept_ids)+1, len(samples), figsize=(20,len(concept_ids)*2+2), facecolor="white")
#     keys = list(dict.keys())
#     Rsum = [np.zeros_like(R) for i in range(len(samples))]
#     for i, key in enumerate(keys):
#         for j, item in enumerate(dict[key]):
#             item = item.permute(1,2,0).cpu().detach().numpy()
#             R = item.sum(axis=-1)
#             R, kwargs = process_lrp_before_imshow(R)
#             axes[i,j].imshow(R, **kwargs)
#     #         print(kwargs)
#             axes[i,0].set_ylabel(str(key)+" "+ str(concept_dict[key]), fontsize=20)
#             Rsum[j] += R * (hidden_sample_list[samples[i]][key]- hidden_sample_mean[key]).item()
#     #         if concept_dict[key]:
#     #             Rsum[j] += R * (hidden_sample_list[samples[i]][key] - hidden_sample_mean[key]).item()/(hidden_sample_list[samples[i]][key] - hidden_sample_mean[key]).item()
#     #         else :
#     #             Rsum[j] += R * (hidden_sample_list[samples[i]][key] - hidden_sample_mean[key]).item()/hidden_sample_mean[key].item()
#     #         Rsum[j] += R
#     for j, Rs in enumerate(Rsum):
#         Rs, kwargs = process_lrp_before_imshow(Rs)
#         axes[-1,j].imshow(Rs,**kwargs)
#     plt.tight_layout()


# def plot_CRP(dict, samples):
#     # === Plot Image and LRP ===
#     fig, axes = plt.subplots(2, len(samples), figsize=(20,4), facecolor="white")
#     for i, index in enumerate(samples):
#     #     img = testset[index][0]
#     #     img = load_image(index)[0]
#         img = x_list[index]
#         img = restore_image(img)
#         axes[0,i].imshow(img)
#         axes[0,i].set_title("{}".format(classes[int(torch.argmax(y_list[index]).item())]))
#     for j in range(len(samples)):
#         item = lrp_list[j]
#         item = item.permute(1,2,0).cpu().detach().numpy()
#         R = item.sum(axis=-1)
#         R, kwargs = process_lrp_before_imshow(R)
#         axes[1,j].imshow(R, **kwargs)

#     axes[0,0].set_ylabel("Image", fontsize=20)
#     axes[1,0].set_ylabel("LRP",  fontsize=20)
#     plt.tight_layout()

#     # === Plot CRP ===
#     fig, axes = plt.subplots(len(concept_ids), len(samples), figsize=(20,len(concept_ids)*2), facecolor="white")
#     keys = list(dict.keys())
#     if(len(concept_ids) == 1):
#         for i, key in enumerate(keys):
#             for j, item in enumerate(dict[key]):
#                 item = item.permute(1,2,0).cpu().detach().numpy()
#                 R = item.sum(axis=-1)
#                 R, kwargs = process_lrp_before_imshow(R)
#                 axes[j].imshow(R, **kwargs)
#                 axes[0].set_ylabel(str(key)+" "+str(concept_dict[key]), fontsize=20)
#     else:
#         for i, key in enumerate(keys):
#             for j, item in enumerate(dict[key]):
#                 item = item.permute(1,2,0).cpu().detach().numpy()
#                 R = item.sum(axis=-1)
#                 R, kwargs = process_lrp_before_imshow(R)
#                 axes[i,j].imshow(R, **kwargs)
#                 axes[i,0].set_ylabel(str(key)+" "concept_dict[key], fontsize=20)
#     plt.tight_layout()