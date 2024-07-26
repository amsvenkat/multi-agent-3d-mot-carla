from collections import defaultdict
import os
import numpy as np
from typing import Union , List

EvalBoxType = Union['DetectionBox', 'TrackingBox']

content = {'a': [{'classn': 'car', 'dimension': [1, 2, 3], 'location_cam': [1, 2, 3], 'rotation': 1, 'score': 1},\
     {'classn': 'car', 'dimension': [1, 2, 3], 'location_cam': [1, 2, 3], 'rotation': 1, 'score': 1}],
           'b': [{'classn': 'peds', 'dimension': [1, 2, 3], 'location_cam': [1, 2, 3], 'rotation': 1, 'score': 1}],
           'c': [{'classn': 'car', 'dimension': [1, 2, 3], 'location_cam': [1, 2, 3], 'rotation': 1, 'score': 1},
                 {'classn': 'car', 'dimension': [1, 2, 3], 'location_cam': [
                     1, 2, 3], 'rotation': 1, 'score': 1},
                 {'classn': 'car', 'dimension': [1, 2, 3], 'location_cam': [1, 2, 3], 'rotation': 1, 'score': 1}]}

class Check:
    def __init__(self):
        self.box = defaultdict(list)
    
    def __getitem__(self, item) -> List[EvalBoxType]:
        return self.box[item]

    @property
    def all(self) -> List[EvalBoxType]:
        """ Returns all EvalBoxes in a list. """
        ab = []
        for sample_token in content.keys():
            ab.extend(self[sample_token])
        return ab
    
    def addboxes(self, key, boxes):
        self.box[key].extend(boxes)
    
    @classmethod
    def deserialize(cls, content):
        eb = Check()
        for key, value in content.items():
            eb.addboxes(key, [Boxx.deserialize(i) for i in value])
        return eb

class Boxx:
    def __init__(self, classn, dimension, location_cam, rotation, score) -> None:
        self.classn = classn
        self.dimension = dimension
        self.location_cam = location_cam
        self.rotation = rotation
        self.score = score

    @classmethod
    def deserialize(cls, box):
        return cls(classn = box['classn'], dimension = box['dimension'], location_cam = box['location_cam'], rotation =box['rotation'], score= box['score'])

# eb = Check()
# eb2 = defaultdict(list)

preds = Check.deserialize(content)
print(preds)

# for key, value in content.items():
#    #print(value)
#     #eb.addboxes(key, [Boxx.deserialize(i) for i in value])
#     eb.addboxes(key, [i for i in value])
#     eb2[key].append(value)

# print(eb)


npos = len([1 for gt_box in preds.all if gt_box.classn == 'car'])
print(npos)
print(len(preds.all))
print((preds.all))
#print(len(eb))
#print(eb2)
