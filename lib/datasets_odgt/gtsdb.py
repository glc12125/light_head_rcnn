# encoding: utf-8
"""
@author: liangchuan gu
@contact: liangchuan.gu@robok.ai
"""

class GTSDBBasic:
    class_names = [
        'speed limit 20 (prohibitory)',
        'speed limit 30 (prohibitory)',
        'speed limit 50 (prohibitory)',
        'speed limit 60 (prohibitory)',
        'speed limit 70 (prohibitory)',
        'speed limit 80 (prohibitory)',
        'restriction ends 80 (other)',
        'speed limit 100 (prohibitory)',
        'speed limit 120 (prohibitory)',
        'no overtaking (prohibitory)',
        'no overtaking (trucks) (prohibitory)',
        'priority at next intersection (danger)',
        'priority road (other)',
        'give way (other)',
        'stop (other)',
        'no traffic both ways (prohibitory)',
        'no trucks (prohibitory)',
        'no entry (other)',
        'danger (danger)',
        'bend left (danger)',
        'bend right (danger)',
        'bend (danger)',
        'uneven road (danger)',
        'slippery road (danger)',
        'road narrows (danger)',
        'construction (danger)',
        'traffic signal (danger)',
        'pedestrian crossing (danger)',
        'school crossing (danger)',
        'cycles crossing (danger)',
        'snow (danger)',
        'animals (danger)',
        'restriction ends (other)',
        'go right (mandatory)',
        'go left (mandatory)',
        'go straight (mandatory)',
        'go right or straight (mandatory)',
        'go left or straight (mandatory)',
        'keep right (mandatory)',
        'keep left (mandatory)',
        'roundabout (mandatory)',
        'restriction ends (overtaking) (other)',
        'restriction ends (overtaking (trucks)) (other)']
    classes_originID = {
        'speed limit 20 (prohibitory)':0,
        'speed limit 30 (prohibitory)':1,
        'speed limit 50 (prohibitory)':2,
        'speed limit 60 (prohibitory)':3,
        'speed limit 70 (prohibitory)':4,
        'speed limit 80 (prohibitory)':5,
        'restriction ends 80 (other)':6,
        'speed limit 100 (prohibitory)':7,
        'speed limit 120 (prohibitory)':8,
        'no overtaking (prohibitory)':9,
        'no overtaking (trucks) (prohibitory)':10,
        'priority at next intersection (danger)':11,
        'priority road (other)':12,
        'give way (other)':13,
        'stop (other)':14,
        'no traffic both ways (prohibitory)':15,
        'no trucks (prohibitory)':16,
        'no entry (other)':17,
        'danger (danger)':18,
        'bend left (danger)':19,
        'bend right (danger)':20,
        'bend (danger)':21,
        'uneven road (danger)':22,
        'slippery road (danger)':23,
        'road narrows (danger)':24,
        'construction (danger)':25,
        'traffic signal (danger)':26,
        'pedestrian crossing (danger)':27,
        'school crossing (danger)':28,
        'cycles crossing (danger)':29,
        'snow (danger)':30,
        'animals (danger)':31,
        'restriction ends (other)':32,
        'go right (mandatory)':33,
        'go left (mandatory)':34,
        'go straight (mandatory)':35,
        'go right or straight (mandatory)':36,
        'go left or straight (mandatory)':37,
        'keep right (mandatory)':38,
        'keep left (mandatory)':39,
        'roundabout (mandatory)':40,
        'restriction ends (overtaking) (other)':41,
        'restriction ends (overtaking (trucks)) (other)':42}
    num_classes = 43


class GTSDB(GTSDBBasic):
    pass
    # train_root_folder = ''
    # train_source = os.path.join(
    #     config.root_dir, 'data', 'MSCOCO/odformat/coco_trainvalmini.odgt')
    # eval_root_folder = ''
    # eval_source = os.path.join(
    #     config.root_dir, 'data', 'MSCOCO/odformat/coco_minival2014.odgt')
    # eval_json = os.path.join(
    #     config.root_dir, 'data', 'MSCOCO/instances_minival2014.json')


if __name__ == "__main__":
    # coco = COCOIns()
    from IPython import embed
    embed()
