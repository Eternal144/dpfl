#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.Nets import CNNMnist
from utils.options import args_parser
import os

def save_model_for_deployment():
    # 创建保存模型的目录
    save_dir = './saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 加载参数
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # 创建模型
    model = CNNMnist(args=args).to(args.device)
    
    # 加载训练好的模型权重
    checkpoint = torch.load('./mylog/best_model.pth', map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    # 准备示例输入
    dummy_input = torch.randn(1, 1, 28, 28).to(args.device)  # MNIST输入大小
    
    # 保存为ONNX格式
    onnx_path = os.path.join(save_dir, 'mnist_cnn.onnx')
    torch.onnx.export(
        model,                  # 要导出的模型
        dummy_input,           # 模型输入
        onnx_path,             # 保存路径
        export_params=True,    # 存储训练好的参数权重
        opset_version=11,      # ONNX算子集版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],     # 输入名
        output_names=['output'],   # 输出名
        dynamic_axes={             # 动态尺寸
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("模型已保存到:", save_dir)
    print("ONNX格式: mnist_cnn.onnx")
    print(f"模型准确率: {checkpoint['accuracy']:.2f}%")
    print(f"保存时的轮次: {checkpoint['epoch']}")

if __name__ == '__main__':
    save_model_for_deployment() 