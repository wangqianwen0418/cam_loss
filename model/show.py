from get_coco import COCOData
import numpy as np
import cv2
import json
from sklearn.manifold import TSNE
data = COCOData(data_dir="../data/coco/", 
                    COI=['cat'], 
                    img_set="train", 
                    year=2017)

allmaps = np.load("../data/cache/allmaps_bbox2017.npy")
preds = np.load("../data/cache/preds_bbox2017.npy")
pred_maps = np.load("../data/cache/predmaps_bbox2017.npy")
true_preds = np.load("../data/cache/true_preds_2017.npy")
# ids = []
# for i in range(preds.shape[0]):
#     if(abs(preds[i][0]-true_preds[i][0])>0.15 ):
#         ids.append(i)
# print("len of low confidence", len(ids))

identity = 255 - np.arange(256, dtype = np.dtype("uint8"))
zeros = np.zeros(256, dtype = np.dtype("uint8"))
ones = np.ones(256, dtype = np.dtype("uint8"))
lut = np.dstack((identity, identity, ones))

tsne = []
pos = TSNE(n_components=2).fit_transform(np.sum(allmaps, axis=(1,2)))
minP = np.min(pos, axis=0)
pos = pos - minP
maxP = np.max(pos, axis=0)
pos = pos/maxP


for id in range(preds.shape[0]):
    img_path = data.img_at(id)
    
    maps = allmaps[id]
    pred = preds[id]
    pred_map = pred_maps[id]
    pred_map = cv2.resize(np.sum(pred_maps[id], axis=2), (7,7))

    pred_map = pred_map - np.min(pred_map)
    pred_map = pred_map / np.max(pred_map) 
    pred_map = np.uint8(255 * pred_map)
    

    img = cv2.imread(img_path)
    (h, w, _) = img.shape
    pred_map = cv2.resize(pred_map, (w, h), interpolation = cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(pred_map, cv2.COLORMAP_OCEAN)
    # pred_map = np.repeat(np.expand_dims(pred_map, axis=-1), repeats=3, axis=2)
    # heatmap = cv2.LUT(pred_map, lut)
    result = cv2.addWeighted(img, 0.3, heatmap, 0.4, 9)
    # cv2.imshow('image', result)
    # cv2.waitKey(500)
    cv2.imwrite('../data/cache/heatmaps_bbox2017/{}.jpg'.format(id), result)

    tsne.append({"pos":pos[id].tolist() , "pred":preds[id].tolist()[0]})


with open("../frontEnd/src/cache/heatmaps_bbox2017/pos.json", "w") as jsonf:
    json.dump(tsne, jsonf)
jsonf.close()

    
    
#     # for i in range(maps.shape[2]):
#     #     map = maps[:,:,i]
#     #     map = cv2.resize(np.sum(pred_maps[id], axis=2), (7,7))
#     #     map = map - np.min(map)
#     #     map = map / np.max(map) 
#     #     map = np.uint8(255 * map)
        
#     #     print(map.shape, np.sum(map))

#     #     # heatmap = cv2.applyColorMap(cv2.resize(map, (w, h), interpolation = cv2.INTER_CUBIC), cv2.COLORMAP_AUTUMN)
#     #     # cv2.putText(heatmap, str(i), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
#     #     # cv2.imshow('image', heatmap)
#     #     # cv2.waitKey(100)




