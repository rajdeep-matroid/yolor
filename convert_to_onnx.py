import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from functools import reduce
import argparse
import onnx_graphsurgeon as gs
import onnx
import tensorrt as trt
import numpy as np
from onnx import shape_inference
from models.models import Darknet
from utils.datasets import CalibratorImages
from torch2trt.calibration import DatasetCalibrator, TensorBatchDataset

def convert_to_engine(onnx_f, im, sparsify=False, int8=False, half=False, int8_calib_dataset=None, 
                      calib_batch_size=4, workspace=28, calib_algo='entropy2', end2end=False, 
                      conf_thres=0.45, iou_thres=0.25, max_det=100):
    prefix = 'TensorRT:'
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << workspace
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(onnx_f)
    
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    
    
    print(f'{prefix} Network Description:')
    for inp in inputs:
        print(f'{prefix}\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
    for out in outputs:
        print(f'{prefix}\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')
        
        
    if end2end:
        previous_output = network.get_output(0)
        network.unmark_output(previous_output)
        strides = trt.Dims([1,1,1])
        starts = trt.Dims([0,0,0])
        bs, num_boxes, temp = previous_output.shape
        shapes = trt.Dims([bs, num_boxes, 4])
        boxes = network.add_slice(previous_output, starts, shapes, strides)
        num_classes = temp  - 5
        starts[2] = 4
        shapes[2] = 1
        obj_score = network.add_slice(previous_output, starts, shapes, strides)
        starts[2] = 5
        shapes[2] = num_classes
        scores = network.add_slice(previous_output, starts, shapes, strides)
        updated_scores = network.add_elementwise(obj_score.get_output(0), scores.get_output(0), trt.ElementWiseOperation.PROD)
        
        registry = trt.get_plugin_registry()
        assert(registry)
        creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
        assert(creator)
        
        fc = []
        fc.append(trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("max_output_boxes", np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
        fc.append(trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
        fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
        
        fc = trt.PluginFieldCollection(fc)
        nms_layer = creator.create_plugin("nms_layer", fc)
        
        layer = network.add_plugin_v2([boxes.get_output(0), updated_scores.get_output(0)], nms_layer)
        layer.get_output(0).name = "num"
        layer.get_output(1).name = "boxes"
        layer.get_output(2).name = "scores"
        layer.get_output(3).name = "classes"
        for i in range(4):
            network.mark_output(layer.get_output(i))
                
    f = onnx_f.replace('onnx','engine')
    print(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine in {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)

    inputs_in = im
    if not isinstance(im, tuple):
        im = (im,)

    if int8:
        if int8_calib_dataset is None:
            int8_calib_dataset = TensorBatchDataset(inputs_in)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if builder.platform_has_fast_int8:
            print(f'{prefix} building INT8 engine in {f}')
            config.set_flag(trt.BuilderFlag.INT8)
        
        if calib_algo=='entropy2':
            algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
        elif calib_algo == 'entropy':
            algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION
        else:
            algo = trt.CalibrationAlgoType.MINMAX_CALIBRATION
        calibrator = DatasetCalibrator(
            im, int8_calib_dataset, batch_size=calib_batch_size, algorithm=algo
        )

        config.int8_calibrator = calibrator
        
    if sparsify:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
        
        
def get_module_by_name(model, access_string):
    names = access_string.split('.')[:-1]
    return reduce(getattr, names, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolor_csp_star.pt', help='weights path')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_csp.cfg', help='config path')
    parser.add_argument('--output', type=str, default='yolor_csp.onnx', help='output ONNX model path')
    parser.add_argument('--max_size', type=int, default=640, help='max size of input image')
    parser.add_argument('--sparsify', default=False, action='store_true', help='enable model sparsification')
    parser.add_argument('--prop', default=0.3, type=float, help='sparsification fraction')
    parser.add_argument('--struct', default=False, action='store_true', help='enable structured sparsification')
    parser.add_argument('--simplify', default=False, action='store_true', help='simplify onnx model')
    parser.add_argument('--half', default=False, action='store_true', help='enable FP16 quantization')
    parser.add_argument('--int8', default=False, action='store_true', help='enable INT8 quantization')
    parser.add_argument('--calibrate', default=False, action='store_true', help='enable INT8 calibration')
    parser.add_argument('--calib-num-images', type=int, default=200, help='number of images to be used for calibration')
    parser.add_argument('--calib-batch-size', type=int, default=4, help='batch size to be used for INT8 calibration')
    parser.add_argument('--workspace', type=int, default=28, help='workspace size')
    parser.add_argument('--opset', type=int, default=12, help='onnx opset')
    parser.add_argument('--seed', type=int, default=10, help='seed for TensorRT INT8 calibration')
    parser.add_argument('--calib-algo', type=str, default='entropy2', choices=['entropy', 'minmax', 'entropy2'], help='calibration algorithm for INT8 optimization')
    parser.add_argument('--trt', default=False, action='store_true', help='enable TRT optimization')
    parser.add_argument('--end2end', default=False, action='store_true', help='include nms plugin')
    parser.add_argument('--conf-thres', default=0.45, type=float, help='confidence threshold')
    parser.add_argument('--iou-thres', default=0.25, type=float, help='iou threshold')
    parser.add_argument('--topk-all', default=100, type=int, help='maximum number of detections')
    parser.add_argument('--device', default='cuda:0', type=str)
    opt = parser.parse_args()
    model_cfg = opt.cfg
    model_weights = opt.weights 
    output_model_path = opt.output
    max_size = opt.max_size

    device = torch.device(opt.device)
    # Load model
    model = Darknet(model_cfg, (max_size, max_size)).cuda()
    model.load_state_dict(torch.load(model_weights, map_location=device)['model'])
    mode = model.eval()
    
    
    # model sparsification
    if opt.sparsify:
        print("Beginning model sparsification")
        modules = [module[0] for module in model.named_parameters()]
        parameters_to_prune = []
        for module in modules:
            if 'weight' in module:
                obj = get_module_by_name(model, module)
                if opt.struct and not isinstance(nn.BatchNorm2d):
                    prune.ln_structured(obj, 'weight', amount=opt.prop, n=1, dim=0)
                    prune.remove(obj, 'weight')
                else:
                    parameters_to_prune.append((obj, 'weight'))
        
        if not opt.struct:
            parameters_to_prune = tuple(parameters_to_prune)
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=opt.prop)
            for param in parameters_to_prune:
                prune.remove(*param)
        print("Model sparsification successful")
        
    
    model.to(device)
    img = torch.zeros((1, 3, max_size, max_size), device=device)  # init img
    if opt.half:
        model = model.half()
    

    print('Convert from Torch to ONNX')
    # Export the model
    torch.onnx.export(model,               # model being run
                      img,                         # model input (or a tuple for multiple inputs)
                      output_model_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=opt.opset,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['images'],   # the model's input names
                      output_names = ['output'], # the models's output names
                      )
    onnx_model = onnx.load(output_model_path)
    onnx.checker.check_model(onnx_model)

    print('Remove unused outputs')
    onnx_module = shape_inference.infer_shapes(onnx.load(output_model_path))
    while len(onnx_module.graph.output) != 1:
        for output in onnx_module.graph.output:
            if output.name != 'output':
                print('--> remove', output.name)
                onnx_module.graph.output.remove(output)
    graph = gs.import_onnx(onnx_module)
    graph.cleanup()
    graph.toposort()
    graph.fold_constants().cleanup()
    onnx.save_model(gs.export_onnx(graph), output_model_path)
    
    if opt.trt:
        print("Convert from ONNX to TensorRT")
        calib_dataset = None
        if opt.int8:
            if opt.calibrate:
                calib_dataset = CalibratorImages('../datasets/coco/val2017/*.jpg', auto=False, num_images=opt.calib_num_images, seed=opt.seed)
                
        convert_to_engine(output_model_path, 
                          img,
                          sparsify=opt.sparsify, 
                          half=opt.half, 
                          int8=opt.int8,  
                          int8_calib_dataset=calib_dataset, 
                          calib_batch_size=opt.calib_batch_size, 
                          workspace=opt.workspace, 
                          calib_algo=opt.calib_algo,
                          end2end=opt.end2end,
                          conf_thres=opt.conf_thres,
                          iou_thres=opt.iou_thres,
                          max_det=opt.topk_all)
        
    print('Conversions successful!')


