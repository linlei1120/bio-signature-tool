"""
演示和测试脚本
用于测试签名提取功能
"""

import cv2
import numpy as np
from signature_extractor import SignatureExtractor
import os


def create_test_signature():
    """
    创建一个测试签名图片
    Returns:
        测试图片的文件路径
    """
    # 创建白色背景
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # 绘制手写风格的签名（模拟）
    # 使用深色笔迹
    color = (0, 0, 0)  # 黑色
    thickness = 3
    
    # 绘制一些曲线模拟签名
    points = [
        (100, 200), (150, 180), (200, 190), (250, 200),
        (300, 180), (350, 170), (400, 180)
    ]
    
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], color, thickness)
    
    # 添加一些装饰线条
    cv2.line(img, (150, 250), (200, 230), color, thickness)
    cv2.line(img, (200, 230), (250, 240), color, thickness)
    
    # 添加点
    cv2.circle(img, (120, 220), 4, color, -1)
    cv2.circle(img, (380, 190), 4, color, -1)
    
    # 保存测试图片
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    test_image_path = os.path.join(test_dir, "test_signature.png")
    cv2.imwrite(test_image_path, img)
    print(f"测试图片已创建: {test_image_path}")
    
    return test_image_path


def test_extraction():
    """测试签名提取功能"""
    print("\n=== 签名提取测试 ===\n")
    
    # 创建测试图片
    test_image = create_test_signature()
    
    # 创建提取器
    extractor = SignatureExtractor()
    
    # 测试提取
    print("正在测试签名提取...")
    signature = extractor.extract_from_file(test_image)
    
    if signature is not None:
        print("\n[成功] 测试成功！")
        print(f"提取的签名大小: {signature.shape[:2]}")
        print(f"提取的签名已保存到: {extractor.output_dir}")
        
        # 显示结果
        original = cv2.imread(test_image)
        
        # 创建白色背景用于显示透明签名
        h, w = signature.shape[:2]
        preview = np.ones((h, w, 3), dtype=np.uint8) * 255
        alpha = signature[:, :, 3] / 255.0
        for c in range(3):
            preview[:, :, c] = (
                alpha * signature[:, :, c] + 
                (1 - alpha) * preview[:, :, c]
            )
        
        # 并排显示
        cv2.imshow('Original Image', original)
        cv2.imshow('Extracted Signature', preview)
        print("\n按任意键关闭预览窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n[失败] 测试失败：未能提取签名")


def quick_start_guide():
    """快速入门指南"""
    print("\n" + "=" * 60)
    print("手写签名提取程序 - 快速入门")
    print("=" * 60)
    print("\n【功能介绍】")
    print("这个程序可以从图片中自动提取手写签名，并生成透明背景的PNG图片。")
    print("\n【三种使用模式】")
    print("1. 图片文件模式 - 从现有图片中提取签名")
    print("2. 实时摄像头模式 - 实时拍照并提取签名")
    print("3. 批量处理模式 - 批量处理多张图片")
    print("\n【使用建议】")
    print("- 使用高对比度图片（深色签名，浅色背景）")
    print("- 保持签名清晰，背景简洁")
    print("- 如果效果不佳，可以调整阈值参数")
    print("\n【快速开始】")
    print("运行主程序: python signature_extractor.py")
    print("运行测试: python demo_signature_test.py")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    quick_start_guide()
    
    response = input("是否运行测试程序？(y/n): ").strip().lower()
    if response == 'y':
        test_extraction()
    else:
        print("\n可以直接运行主程序:")
        print("  python signature_extractor.py")

