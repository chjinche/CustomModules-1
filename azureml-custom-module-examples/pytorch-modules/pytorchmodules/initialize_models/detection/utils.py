import torch


def remove_images_without_annotations(dataset, cat_list=None):
    # def _has_only_empty_bbox(anno):
    #     return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)
    def _has_only_empty_bbox(boxes):
        return all(any((o - box[i]) <= 1 for i, o in enumerate(box[2:])) for box in boxes)

    # def _count_visible_keypoints(anno):
    #     return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    # min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        boxes = anno['boxes']
        if boxes.nelement() == 0:
            return False
        if _has_only_empty_bbox(boxes):
            return False
        # if it's empty, there is no annotation
        # if len(anno) == 0:
        #     return False
        # # if all boxes have close to zero area, there is no annotation
        # if _has_only_empty_bbox(anno):
        #     return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        # if "keypoints" not in anno[0]:
        #     return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        # if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        #     return True
        # return False
        return True

    # assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    exclude_ids = []
    for ds_idx, (image, target) in enumerate(dataset):
        # print(f'ds_idx {ds_idx}')
        if _has_valid_annotation(target):
            ids.append(ds_idx)
        else:
            exclude_ids.append(target['image_id'])

    # ids = []
    # for ds_idx, img_id in enumerate(dataset.ids):
    #     ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
    #     anno = dataset.coco.loadAnns(ann_ids)
    #     if cat_list:
    #         anno = [obj for obj in anno if obj["category_id"] in cat_list]
    #     if _has_valid_annotation(anno):
    #         ids.append(ds_idx)
    print(f'invalid image id: {exclude_ids}')
    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset