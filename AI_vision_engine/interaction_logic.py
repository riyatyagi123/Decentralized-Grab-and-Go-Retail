import cv2
class SpatialInteractionEngine:
    def __init__(self, iou_threshold=0.05):
        self.iou_threshold = iou_threshold
        self.shelf_zone = None
    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    def determine_interaction_state(self, hand_box, product_box, product_in_shelf_zone):
        iou = self.calculate_iou(hand_box, product_box)
        state = "Idle"
        if iou > self.iou_threshold:
            state = "Touch"
            if not product_in_shelf_zone:
                state = "Take"
        elif product_in_shelf_zone and state == "Take":
            state = "Return"
        return state, iou
