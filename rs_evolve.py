import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import cv2
import json
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import os.path as osp
#from meters import average_accuracy
from sklearn.metrics import accuracy_score
from torchvision.utils import save_image, make_grid

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def visualize_single_attention(attn_tensor, save_path,image_id, mask_dir=None):
    if isinstance(attn_tensor, np.ndarray):
        attn_map = attn_tensor
    else:
        attn_map = attn_tensor.cpu().numpy()
    original_save_path = os.path.splitext(save_path)[0] + '_original.npy'
    np.save(original_save_path, attn_map)
    attn_norm = (attn_map * 255).astype(np.uint8)
    if mask_dir is not None:
        mask_path = os.path.join(mask_dir, f"{image_id:05d}.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            h, w = mask.shape
            attn_resized = cv2.resize(attn_norm, (w, h), interpolation=cv2.INTER_LINEAR)

            attn_heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)

            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            alpha = 0.6
            overlay = cv2.addWeighted(mask_rgb, 1 - alpha, attn_heatmap, alpha, 0)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            cv2.imwrite(save_path, overlay)
        else:
            img_pil = Image.fromarray(attn_norm).convert('L')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img_pil.save(save_path)
    else:
        img_pil = Image.fromarray(attn_norm).convert('L')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img_pil.save(save_path)

def self_enhanced_fun(self_attn,cross_attn_ori,res,densecrf=False,img=None,beta=0.4,K_k=200,tagid = 1,imageid=100):
    if self_attn.size() < cross_attn_ori.size():
        self_attn = F.interpolate(self_attn.reshape(1, 1, self_attn.shape[0] ** 2, self_attn.shape[0] ** 2),
                                  size=(res ** 2, res ** 2), mode='bilinear').reshape(res, res, res, res)
    avg_self_attn = torch.zeros_like(cross_attn_ori)
    H, W = cross_attn_ori.shape
    for i in range(H):
        for j in range(W):
            tmp1 = self_attn[int(i), int(j)]
            tmp2 = cross_attn_ori[int(i), int(j)] * tmp1
            avg_self_attn += tmp2
    avg_self_attn = avg_self_attn / (avg_self_attn.norm(p=2) + 1e-6)
    avg_self_attn = avg_self_attn - avg_self_attn.min()
    avg_self_attn = avg_self_attn / (avg_self_attn.max() + 1e-6)
    return avg_self_attn


def region_grow(prediction_map, weighted_attn):
    H, W = prediction_map.shape
    visited = torch.zeros_like(prediction_map, dtype=torch.bool)
    region_mask = torch.zeros_like(prediction_map, dtype=torch.uint8)
    threshold = 0.3
    flat_attn = weighted_attn.flatten()
    unique_vals = torch.unique(flat_attn)
    unique_vals, _ = torch.sort(unique_vals, descending=True)
    top_vals = unique_vals[:7]
    seed_indices = []
    for val in top_vals:
        indices = (weighted_attn == val).nonzero(as_tuple=False)
        seed_indices.extend([tuple(idx.tolist()) for idx in indices])
    queue = []
    for y, x in seed_indices:
        queue.append((y, x))
        visited[y, x] = True
        region_mask[y, x] = 1
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1), (1, 0),  (1, 1)]
    while queue:
        y, x = queue.pop(0)
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if not visited[ny, nx] and prediction_map[ny, nx] > threshold:
                    queue.append((ny, nx))
                    visited[ny, nx] = True
                    region_mask[ny, nx] = 1
    return region_mask


def mask_evaluate_all(pred, gt):
    assert pred.shape[-2:] == gt.shape[-2:]
    temp = pred * gt
    inter = temp.sum()
    union = ((pred + gt) - temp).sum()
    iou = inter / (union + 1e-6)
    TP = (pred * gt).sum().float()
    FP = (pred * (1 - gt)).sum().float()
    FN = ((1 - pred) * gt).sum().float()
    TN = ((1 - pred) * (1 - gt)).sum().float()
    # TP = np.logical_and(pred == 1, gt == 1).sum()
    # TN = np.logical_and(pred == 0, gt == 0).sum()
    # FP = np.logical_and(pred == 1, gt == 0).sum()
    # FN = np.logical_and(pred == 0, gt == 1).sum()
    #return inter, union, iou, OA, AA,Recall,Precision,F1,Kappa
    return inter, union, iou,TP,TN,FP,FN


def box_evaluate_all(pred_mask, gt_box):
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().detach()
    if pred_mask.dim() == 3:
        pred_mask = pred_mask[0]
    pred_mask = pred_mask.bool()

    if pred_mask.any():
        cols = pred_mask.any(dim=0)
        rows = pred_mask.any(dim=1)

        xmin = cols.nonzero(as_tuple=True)[0][0].item()
        xmax = cols.nonzero(as_tuple=True)[0][-1].item()
        ymin = rows.nonzero(as_tuple=True)[0][0].item()
        ymax = rows.nonzero(as_tuple=True)[0][-1].item()

        box_pred = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    else:
        box_pred = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0}

    box_gt = {
        "xmin": int(gt_box[0]),
        "ymin": int(gt_box[1]),
        "xmax": int(gt_box[2]),
        "ymax": int(gt_box[3]),
    }
    xA = max(box_pred["xmin"], box_gt["xmin"])
    yA = max(box_pred["ymin"], box_gt["ymin"])
    xB = min(box_pred["xmax"], box_gt["xmax"])
    yB = min(box_pred["ymax"], box_gt["ymax"])

    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    predW = box_pred["xmax"] - box_pred["xmin"] + 1
    predH = box_pred["ymax"] - box_pred["ymin"] + 1
    predArea = predW * predH

    gtW = box_gt["xmax"] - box_gt["xmin"] + 1
    gtH = box_gt["ymax"] - box_gt["ymin"] + 1
    gtArea = gtW * gtH

    unionArea = predArea + gtArea - interArea
    iou = interArea / (unionArea + 1e-6)

    return interArea, unionArea, iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_result(llm_save_pathresult, pred_mask):
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    binary_mask = (pred_mask_np * 255).astype(np.uint8)
    mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    y_coords, x_coords = np.where(pred_mask_np > 0)

    if len(x_coords) > 0 and len(y_coords) > 0:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        cv2.rectangle(mask_bgr, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    cv2.imwrite(llm_save_pathresult, mask_bgr)


def dist_evaluate_self_enhanced():
    self_res = 32
    inter_res = self_res
    h,w = 800,800
    thresholds = [0.4]
    results = {t: {'inter': [], 'union': [], 'iou': []} for t in thresholds}
    results_box = {t: {'inter': [], 'union': [], 'iou': []} for t in thresholds}

    png_data = json.load(open("./data/rrsisd.json"))
    png_data = png_data["test"]

    for idx, data in tqdm(enumerate(png_data), total=len(png_data)):
        img_idx = data['iid']
        mask_path = osp.join(
            "./data/targetmask",
            "{:05d}.png".format(int(img_idx)),
        )
        mask_target = torch.from_numpy(np.array(Image.open(mask_path)) / 255.0).float().to(device)

        gt_box = data['bbox']
        #gt_box = data['bboxes'][0]
        try:
            self_attn = torch.load(f'./outputs/attn_db/{idx}/self_{self_res}.pt', map_location=device)
            self_attn64 = torch.load(f'./outputs/attn_db/{idx}/self_{64}.pt', map_location=device)
            self_attn = self_attn.reshape(1024, 1024)
            self_attn64 = self_attn64.reshape(4096,4096)
            self_attn64 = F.interpolate(self_attn64[None, None, ...], (1024, 1024), mode='bilinear')[0, 0].float()
            self_attn = (self_attn + self_attn64 )/2
        except Exception as e:
            print(f"error: {e}")
            continue
        self_attn = self_attn.reshape(self_res,self_res, self_res, self_res)
        weighted_attn = np.load(f'./llmweight/{img_idx}.npy')
        if isinstance(weighted_attn, np.ndarray):
            weighted_attn = torch.tensor(weighted_attn, dtype=torch.float32).to(device)
        weighted_attn = weighted_attn[0]
        weighted_attn = F.interpolate(weighted_attn[None, None, ...], size=(inter_res, inter_res), mode='bilinear')[0, 0]
        weighted_attn = (weighted_attn - weighted_attn.min()) / (weighted_attn.max() + 1e-6)
        predictions = self_enhanced_fun(self_attn, weighted_attn, inter_res, False, None, tagid=idx, imageid=img_idx)
        predictions_cpu = predictions.detach().cpu()
        weighted_attn_cpu = weighted_attn.detach().cpu()
        mask = region_grow(predictions_cpu, weighted_attn_cpu)
        mask = mask.to(device)
        predictions = mask * predictions
        predictions = F.interpolate(predictions[None,None, ...], (h, w), mode='bilinear')[0,0].float()
        for threshold in thresholds:
            pred_mask = (predictions > threshold).float()
            inter, uninon, instance_iou, tp1, tn1, fp1, fn1 = mask_evaluate_all(pred_mask, mask_target)
            results[threshold]['inter'].append(inter.cpu().item())
            results[threshold]['union'].append(uninon.cpu().item())
            results[threshold]['iou'].append(instance_iou.cpu().item())
            boxinter, boxuninon, boxinstance_iou = box_evaluate_all(pred_mask, gt_box)
            results_box[threshold]['inter'].append(boxinter)
            results_box[threshold]['union'].append(boxuninon)
            results_box[threshold]['iou'].append(boxinstance_iou)
    for threshold in thresholds:
        interss = results[threshold]['inter']
        unionss = results[threshold]['union']
        ious = results[threshold]['iou']

        if len(ious) == 0:
            miou = oiou = acc = 0.0
        else:
            miou = np.mean(ious)
            oiou = sum(interss) / sum(unionss)
            accs_dict = {}
            for thres in [0.3, 0.5,0.7]:
                pred = (np.array(ious) > thres).astype(int)
                acc_val = accuracy_score(np.ones(len(ious)), pred)
                accs_dict[f'acc@{thres:.1f}'] = acc_val

        line = f'Threshold: {threshold:.1f} | mIoU: {miou * 100:.2f}% | oIoU: {oiou * 100:.2f}% | '
        line += ' | '.join([f'{k}: {v * 100:.2f}%' for k, v in accs_dict.items()])
        print(line.strip())

        print("=" * 80)
        print("box")

        interss = results_box[threshold]['inter']
        unionss = results_box[threshold]['union']
        ious = results_box[threshold]['iou']

        if len(ious) == 0:
            miou = oiou = acc = 0.0
        else:
            miou = np.mean(ious)
            oiou = sum(interss) / sum(unionss)
            accs_dict = {}
            for thres in [0.3, 0.5,0.7]:
                pred = (np.array(ious) > thres).astype(int)
                acc_val = accuracy_score(np.ones(len(ious)), pred)
                accs_dict[f'acc@{thres:.1f}'] = acc_val

        line = f'Threshold: {threshold:.1f} | mIoU: {miou * 100:.2f}% | oIoU: {oiou * 100:.2f}% | '
        line += ' | '.join([f'{k}: {v * 100:.2f}%' for k, v in accs_dict.items()])
        print(line.strip())
        print()


def sam_refine_with_mask_tensor(pred_mask_tensor=None, image_np=None, sam_model: SamPredictor=None,yuanshi_copy =None,num_iters=1):
    sam_model.set_image(image_np)

    prev_mask = pred_mask_tensor.clone()
    temp_1 = 0

    for iter_idx in range(num_iters):
        pred_mask = (prev_mask > 0).cpu().numpy().astype(np.uint8)
        if yuanshi_copy is not  None:
            yuanshi_np = yuanshi_copy.cpu().numpy()

        ys, xs = np.where(pred_mask == 1)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, logits = sam_model.predict(
            box=input_box,
            multimask_output=True
        )

        best_mask_np = masks[np.argmax(scores)]
        best_mask = torch.from_numpy(best_mask_np).float().to(prev_mask.device)
        if best_mask.shape != (800, 800):
            best_mask = F.interpolate(
                best_mask[None, None, ...],
                size=(800, 800),
                mode='nearest'
            )[0, 0]

        prev_mask = best_mask

    return best_mask,temp_1


def dist_evaluate_sam_enhanced():
    sam_checkpoint = "./sam_checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).cuda()
    sam_predictor = SamPredictor(sam)

    self_res = 32
    inter_res = self_res

    thresholds = [0]
    results = {t: {'inter': [], 'union': [], 'iou': []} for t in thresholds}
    results_box = {t: {'inter': [], 'union': [], 'iou': []} for t in thresholds}

    png_data = json.load(open("./data/rrsisd.json"))
    png_data = png_data["test"]

    for idx, data in tqdm(enumerate(png_data), total=len(png_data)):
        img_idx = data['iid']
        mask_path = osp.join(
            "./data/targetmask",
            "{:05d}.png".format(int(img_idx)),
        )
        mask_target = torch.from_numpy(np.array(Image.open(mask_path)) / 255.0).float().to(device)

        gt_box = data['bbox']
        try:
            self_attn = torch.load(f'./outputs/attn_db{idx}/self_{self_res}.pt', map_location=device)
            self_attn64 = torch.load(f'./outputs/attn_db{idx}/self_{64}.pt', map_location=device)
            self_attn = self_attn.reshape(1024, 1024)
            self_attn64 = self_attn64.reshape(4096,4096)
            self_attn64 = F.interpolate(self_attn64[None, None, ...], (1024, 1024), mode='bilinear')[0, 0].float()
            self_attn = (self_attn + self_attn64 )/2
        except Exception as e:
            print(f"error： {e}")
            continue
        self_attn = self_attn.reshape(self_res, self_res, self_res, self_res)
        weighted_attn = np.load(f'./llmweight/{img_idx}.npy')
        if isinstance(weighted_attn, np.ndarray):
            weighted_attn = torch.tensor(weighted_attn, dtype=torch.float32).to(device)
        weighted_attn = weighted_attn[0]
        weighted_attn = F.interpolate(weighted_attn[None, None, ...], size=(inter_res, inter_res), mode='bilinear')[0, 0]
        weighted_attn = (weighted_attn - weighted_attn.min()) / (weighted_attn.max() + 1e-6)
        predictions = self_enhanced_fun(self_attn, weighted_attn, inter_res, False, None, tagid=idx, imageid=img_idx)
        predictions_cpu = predictions.detach().cpu()
        weighted_attn_cpu = weighted_attn.detach().cpu()
        mask = region_grow(predictions_cpu, weighted_attn_cpu)
        mask = mask.to(device)
        predictions = mask * predictions
        predictions = F.interpolate(predictions[None,None, ...], (800, 800), mode='bilinear')[0,0].float()
        interploate_predictions_copy = predictions
        predictions = (predictions > 0.4).float()
        image_path = osp.join("./data/images", "{:05d}.jpg".format(int(img_idx)))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = image
        sam_refine_predictions,temp = sam_refine_with_mask_tensor(predictions,  image_np, sam_predictor,yuanshi_copy=interploate_predictions_copy,num_iters=1)

        for threshold in thresholds:
            pred_mask = (sam_refine_predictions > threshold).float()
            inter, uninon, instance_iou, tp1, tn1, fp1, fn1 = mask_evaluate_all(pred_mask, mask_target)
            results[threshold]['inter'].append(inter.cpu().item())
            results[threshold]['union'].append(uninon.cpu().item())
            results[threshold]['iou'].append(instance_iou.cpu().item())

            boxinter, boxuninon, boxinstance_iou = box_evaluate_all(pred_mask, gt_box)
            results_box[threshold]['inter'].append(boxinter)
            results_box[threshold]['union'].append(boxuninon)
            results_box[threshold]['iou'].append(boxinstance_iou)

    for threshold in thresholds:
        interss = results[threshold]['inter']
        unionss = results[threshold]['union']
        ious = results[threshold]['iou']

        if len(ious) == 0:
            miou = oiou = acc = 0.0
        else:
            miou = np.mean(ious)
            oiou = sum(interss) / sum(unionss)
            accs_dict = {}
            for thres in [0.3, 0.5,0.7]:
                pred = (np.array(ious) > thres).astype(int)
                acc_val = accuracy_score(np.ones(len(ious)), pred)
                accs_dict[f'acc@{thres:.1f}'] = acc_val

        line = f'Threshold: {threshold:.1f} | mIoU: {miou * 100:.2f}% | oIoU: {oiou * 100:.2f}% | '
        line += ' | '.join([f'{k}: {v * 100:.2f}%' for k, v in accs_dict.items()])
        print(line.strip())

        print("=" * 80)
        print("box")

        interss = results_box[threshold]['inter']
        unionss = results_box[threshold]['union']
        ious = results_box[threshold]['iou']

        if len(ious) == 0:
            miou = oiou = acc = 0.0
        else:
            miou = np.mean(ious)
            oiou = sum(interss) / sum(unionss)
            accs_dict = {}
            for thres in [0.3, 0.5, 0.7]:
                pred = (np.array(ious) > thres).astype(int)
                acc_val = accuracy_score(np.ones(len(ious)), pred)
                accs_dict[f'acc@{thres:.1f}'] = acc_val

        line = f'Threshold: {threshold:.1f} | mIoU: {miou * 100:.2f}% | oIoU: {oiou * 100:.2f}% | '
        line += ' | '.join([f'{k}: {v * 100:.2f}%' for k, v in accs_dict.items()])
        print(line.strip())
        print()

if __name__ == '__main__':
    dist_evaluate_self_enhanced()
    #dist_evaluate_sam_enhanced()        #This decision determines whether to apply the final SAM refinement.




