"""P2-aware STAL for YOLO26 (reported as an ablation finding, not used by the final model).

YOLO26's small-target-aware label assignment enlarges tiny ground-truth boxes to a floor equal to the
second detection stride before selecting anchor candidates. With P3-P5 heads that floor is 16 px; once a
P2 head is added the second stride becomes 8 and the floor silently halves. `stal_floor_callback(16)`
restores the 16 px floor. Register it with `model.add_callback("on_train_batch_start", ...)`.
"""
from ultralytics.utils.torch_utils import unwrap_model


def stal_floor_callback(floor=16):
    def _set(trainer):
        m = unwrap_model(trainer.model)
        if getattr(m, "_stal_floor_set", False):
            return
        if getattr(m, "criterion", None) is None:
            m.criterion = m.init_criterion()
        c = m.criterion
        assert hasattr(c, "one2many"), "expects an end-to-end (YOLO26) criterion"
        for loss in (c.one2many, c.one2one):
            loss.assigner.stride_val = floor
        m._stal_floor_set = True
    return _set
