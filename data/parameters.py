import os

params = {}

# general
# params["CKPT_PATH"] = "./checkpoint/31Aug7pm/"
# if not os.path.exists(params["CKPT_PATH"]):
#     os.mkdir(params["CKPT_PATH"])

params["num_classes"] = 1
params["num_anchors"] = 9
params["input_size"] = 300
params["batch_size"] = 4

# encoder.py
params["anchor_areas"] = [8. * 8., 16 * 16., 32 * 32., 64 * 64., 128 * 128]
# params["anchor_areas"] = [5. * 5., 10 * 10., 15. * 15., 20. * 20., 25. * 25.]

params["aspect_ratios"] = [1 / 2., 1 / 1., 2 / 1.]
params["scale_ratios"] = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]

params["IOU_THRESH"] = 0.2
params["CLS_THRESH"] = 0.5
params["NMS_THRESH"] = 0.001


# loss.py
params["alpha"] = 0.25
params["gamma"] = 2
