from pycocotools.coco import COCO 
import os
import numpy as np

from keras.preprocessing.image import load_img, img_to_array

class COCOData(object):
    def __init__(self, data_dir, COI=['cat'], img_set='train', batch_size=32, target_size=(224, 224)):
        if img_set not in ['train', 'val', 'test']:
            raise ValueError('img_set should be neither `train`, `val` or `test` ') 
        annFile = os.path.join(data_dir, 'annotations/instances_{}2017.json'.format(img_set))

        self.data_dir = data_dir
        self.img_set = img_set
        self.coco = COCO(annFile)
        self.name = 'coco_'+img_set
        self.catNms = COI
        self.catIds =self.coco.getCatIds(catNms=self.catNms)
        # self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.imgIds = self.gt_img_ids()
        self.imgs = self.coco.loadImgs(self.imgIds)
        self.annIds = self.coco.getAnnIds(imgIds=self.imgIds, iscrowd=0)

        self.batch_size = batch_size

        self.num_images = len(self.imgIds)
        self.data_y = self.gt_bag_labels()
        self.cur = 0
        self.perm = np.random.permutation(np.arange(self.num_images))
        self.target_size = target_size

        

    def gt_img_ids(self):
        # hack for now, pos: cat&chiar or cat&couch, neg: only chair or couch
        negIds = self.coco.getCatIds(catNms=['chair', 'couch'])
        pos_1 = self.coco.getImgIds(catIds=self.catIds +[negIds[0]])
        pos_2 = self.coco.getImgIds(catIds=self.catIds + [negIds[1]])
        pos_samples = pos_1
        for sample in pos_2:
            if sample not in pos_samples:
                pos_samples.append(sample)

        neg_1 = self.coco.getImgIds(catIds=[negIds[0]])
        neg_samples = []
        i = 0
        while i < len(pos_samples)//2:
            sample = neg_1.pop()
            if (sample not in pos_samples) and (sample not in neg_samples):
                i += 1
                neg_samples.append(sample)
        neg_2 = self.coco.getImgIds(catIds=[negIds[1]])
        i = 0
        while i < len(pos_samples)//2:
            sample = neg_2.pop()
            if (sample not in pos_samples) and (sample not in neg_samples):
                i += 1
                neg_samples.append(sample)
        return pos_samples + neg_samples



    def gt_bag_labels(self):
        bag_labels = np.zeros((len(self.imgIds),len(self.catIds)+1))
        for i, img in enumerate(self.imgs):
            imgId = img['id']
            annIds = self.coco.getAnnIds(imgIds=imgId)
            anns = self.coco.loadAnns(annIds)
            for ann in  anns:
                catId = ann['category_id']
                try:
                    j = self.catIds.index(catId)
                    print("j", j)
                    bag_labels[i, j+1] = 1 # bag_labels[i:0]=1, none of the cats interested exists
                except ValueError:
                    pass
        return bag_labels

    def imgarr_at(self, ind, target_size):
        """
        #return:
        x: a numpy array of an image
        """
        # if interpolation not in [ "nearest", "bilinear", "bicubic"]:
        #     raise ValueError("Supported methods are `nearest`, `bilinear`, and `bicubic`.")
        img = self.imgs[ind]
        fpath = os.path.join('{}/{}2017'.format(self.data_dir, self.img_set), img['file_name']) 
        img = load_img(fpath, grayscale=False, target_size=target_size)
        x = img_to_array(img, data_format="channels_last") # channels_last for tf, channels_first for theano
        return x

    def gt_bbox_at(self, ind):
        """
        # return:
        bbox: bbox in the form of an numpy array
        """
        imgId = self.imgIds[ind]
        img = self.coco.loadImgs([imgId])[0]
        annId = self.coco.getAnnIds(imgId)
        anns = self.coco.loadAnns(annId)
        bbox = np.zeros(self.target_size)
        neg_sample = True
        for ann in anns:
            if ann['category_id'] in self.catIds:
                neg_sample = False
                x, y, w, h = ann['bbox']
                scale = [self.target_size[0]/img['width'], self.target_size[1]/img['height']]
                x *= scale[0]
                w *= scale[0]
                y *= scale[1]
                h *= scale[1]
                bbox[int(x):int(x+w), int(y):int(y+h)] = 1
        if neg_sample:
            bbox[:, :] = 1
        return bbox
        
        
    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self.perm = np.random.permutation(np.arange(self.num_images))
        self.cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self.cur + self.batch_size >= self.num_images:
            self._shuffle_roidb_inds()
        db_inds = self.perm[self.cur:self.cur + self.batch_size]
        self.cur += self.batch_size
        return db_inds

    def _get_next_minibatch(self, bbox):
        """
        Retrun:
            batch_x: num_bag x 224(h) x 224(w) x 3(c)
            batch_y: num_bag x 80
        """
        batch_x = np.zeros((self.batch_size,) + self.target_size + (3,) , dtype=np.float32)
        batch_y = np.zeros((self.batch_size, len(self.catIds)+1), dtype=np.int16)
        if bbox:
            batch_bbox = np.zeros((self.batch_size,) + self.target_size , dtype=np.float32)
        db_inds = self._get_next_minibatch_inds()
        for i, db_ind in enumerate(db_inds):
            x = self.imgarr_at(db_ind, self.target_size)
            # x = self.image_data_generator.random_transform(x) # implement data agument later
            # x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = self.data_y[db_ind]
            if bbox:
                batch_bbox[i] = self.gt_bbox_at(db_ind)
        if bbox:
            return batch_x, [batch_y, batch_bbox]
        else:
            return batch_x, batch_y

    def get_data(self, start, end):
        end = min(end, self.num_images)
        num_bag = end - start
        data_y = self.data_y()[start: end]
        data_x = np.zeros((num_bag, 3, 224, 224))
        for i in range(start, end):
            im = self.image_at(i)
            blob = im_to_blob(im)
            data_x[i - start] = blob
        return data_x, data_y

    def generate(self, bbox=True):
        """
        This function is used for keras.Model.fit_generator
        """
        self.cur = 0
        while 1:
            x, y = self._get_next_minibatch(bbox)
            yield (x, y)
    
    def next(self, bbox=True):
        x, y = self._get_next_minibatch(bbox)
        return (x, y)

if __name__ == '__main__':
    coco = COCOData(data_dir="../data/coco/", COI=['cat'], img_set="train", batch_size=32, target_size=(224, 224))

    for i in range(4):
        (x, y) = coco.next()
        print(x.shape, y)

