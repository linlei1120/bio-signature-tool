"""
手写签名提取程序
支持从图片中提取签名，并支持实时摄像头拍照
"""

import cv2
import numpy as np
from pathlib import Path
import os
from datetime import datetime


class SignatureExtractor:
    """签名提取器类"""
    
    def __init__(self):
        self.output_dir = "extracted_signatures"
        self._create_output_directory()
    
    def _create_output_directory(self):
        """创建输出目录"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"已创建输出目录: {self.output_dir}")
    
    def preprocess_image(self, image):
        """
        预处理图像
        Args:
            image: 输入的BGR图像
        Returns:
            处理后的图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def extract_signature(self, image, threshold_value=None):
        """
        从图像中提取签名
        Args:
            image: 输入的BGR图像
            threshold_value: 阈值（None表示自动）
        Returns:
            提取的签名图像（带透明通道）
        """
        # 预处理
        processed = self.preprocess_image(image)
        
        # 自适应阈值或固定阈值
        if threshold_value is None:
            # 使用自适应阈值
            binary = cv2.adaptiveThreshold(
                processed, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                21, 10
            )
        else:
            # 使用固定阈值
            _, binary = cv2.threshold(
                processed, threshold_value, 255, 
                cv2.THRESH_BINARY_INV
            )
        
        # 形态学操作：去除噪声
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            print("警告: 未检测到签名")
            return None
        
        # 过滤小轮廓（噪声）
        min_area = 100  # 最小面积阈值
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not valid_contours:
            print("警告: 未检测到有效签名")
            return None
        
        # 获取所有有效轮廓的边界框
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # 添加边距
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        y_max = min(image.shape[0], y_max + margin)
        
        # 裁剪签名区域
        signature_region = binary[y_min:y_max, x_min:x_max]
        original_region = image[y_min:y_max, x_min:x_max]
        
        # 创建带透明通道的图像
        # 将二值图像作为alpha通道
        b, g, r = cv2.split(original_region)
        
        # 反转二值图像（使签名为不透明，背景为透明）
        alpha = signature_region.copy()
        
        # 合并通道创建BGRA图像
        signature_rgba = cv2.merge([b, g, r, alpha])
        
        return signature_rgba
    
    def save_signature(self, signature, filename=None):
        """
        保存签名图像
        Args:
            signature: 签名图像（BGRA格式）
            filename: 保存的文件名（None表示自动生成）
        Returns:
            保存的文件路径
        """
        if signature is None:
            print("错误: 无法保存空的签名图像")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signature_{timestamp}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, signature)
        print(f"签名已保存到: {filepath}")
        return filepath
    
    def extract_from_file(self, image_path, threshold_value=None):
        """
        从文件中提取签名
        Args:
            image_path: 图像文件路径
            threshold_value: 阈值
        Returns:
            提取的签名图像
        """
        if not os.path.exists(image_path):
            print(f"错误: 文件不存在 - {image_path}")
            return None
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像 - {image_path}")
            return None
        
        print(f"正在处理图像: {image_path}")
        
        # 提取签名
        signature = self.extract_signature(image, threshold_value)
        
        if signature is not None:
            # 保存签名
            original_name = Path(image_path).stem
            filename = f"{original_name}_signature.png"
            self.save_signature(signature, filename)
        
        return signature
    
    def realtime_capture(self, camera_index=0):
        """
        实时摄像头拍照提取签名
        Args:
            camera_index: 摄像头索引（默认0为主摄像头）
        """
        print("\n=== 实时摄像头签名提取模式 ===")
        print("操作说明:")
        print("  按 'c' 或 '空格' - 拍照并提取签名")
        print("  按 's' - 保存当前提取的签名")
        print("  按 'q' 或 'ESC' - 退出")
        print("  按 '+' - 增加阈值（使签名更细）")
        print("  按 '-' - 减少阈值（使签名更粗）")
        print("================================\n")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        current_signature = None
        current_frame = None
        threshold_value = 127
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头画面")
                break
            
            # 显示原始画面
            display_frame = frame.copy()
            
            # 在画面上显示提示信息
            cv2.putText(display_frame, "Press 'c' or 'Space' to capture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Threshold: {threshold_value} (+/-)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera - Signature Extraction', display_frame)
            
            # 如果已提取签名，显示预览
            if current_signature is not None:
                # 创建白色背景用于显示
                h, w = current_signature.shape[:2]
                preview = np.ones((h, w, 3), dtype=np.uint8) * 255
                
                # 将BGRA图像叠加到白色背景上
                alpha = current_signature[:, :, 3] / 255.0
                for c in range(3):
                    preview[:, :, c] = (
                        alpha * current_signature[:, :, c] + 
                        (1 - alpha) * preview[:, :, c]
                    )
                
                cv2.imshow('Extracted Signature (Press s to save)', preview)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' 或 ESC
                break
            elif key == ord('c') or key == 32:  # 'c' 或 空格
                print("拍照中...")
                current_frame = frame.copy()
                current_signature = self.extract_signature(current_frame, threshold_value)
                if current_signature is not None:
                    print("签名提取成功！按 's' 保存")
                else:
                    print("未检测到签名，请重试")
            elif key == ord('s'):  # 's'
                if current_signature is not None:
                    self.save_signature(current_signature)
                else:
                    print("没有可保存的签名，请先拍照")
            elif key == ord('+') or key == ord('='):
                threshold_value = min(255, threshold_value + 10)
                print(f"阈值增加到: {threshold_value}")
                if current_frame is not None:
                    current_signature = self.extract_signature(current_frame, threshold_value)
            elif key == ord('-') or key == ord('_'):
                threshold_value = max(0, threshold_value - 10)
                print(f"阈值减少到: {threshold_value}")
                if current_frame is not None:
                    current_signature = self.extract_signature(current_frame, threshold_value)
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头已关闭")


def main():
    """主函数"""
    extractor = SignatureExtractor()
    
    print("=" * 50)
    print("手写签名提取程序")
    print("=" * 50)
    print("\n请选择模式:")
    print("1. 从图片文件提取签名")
    print("2. 实时摄像头拍照提取签名")
    print("3. 批量处理文件夹中的图片")
    print("0. 退出")
    print("-" * 50)
    
    while True:
        choice = input("\n请输入选项 (0-3): ").strip()
        
        if choice == '0':
            print("程序已退出")
            break
        
        elif choice == '1':
            image_path = input("请输入图片文件路径: ").strip().strip('"').strip("'")
            
            # 询问是否使用自定义阈值
            use_custom = input("是否使用自定义阈值？(y/n, 默认n): ").strip().lower()
            threshold = None
            if use_custom == 'y':
                try:
                    threshold = int(input("请输入阈值 (0-255, 推荐127): "))
                except ValueError:
                    print("无效输入，使用自动阈值")
            
            signature = extractor.extract_from_file(image_path, threshold)
            
            if signature is not None:
                print("[成功] 签名提取成功！")
                # 显示预览
                h, w = signature.shape[:2]
                preview = np.ones((h, w, 3), dtype=np.uint8) * 255
                alpha = signature[:, :, 3] / 255.0
                for c in range(3):
                    preview[:, :, c] = (
                        alpha * signature[:, :, c] + 
                        (1 - alpha) * preview[:, :, c]
                    )
                cv2.imshow('Extracted Signature (Press any key to close)', preview)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        elif choice == '2':
            extractor.realtime_capture()
        
        elif choice == '3':
            folder_path = input("请输入文件夹路径: ").strip().strip('"').strip("'")
            
            if not os.path.exists(folder_path):
                print(f"错误: 文件夹不存在 - {folder_path}")
                continue
            
            # 支持的图片格式
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            
            # 获取所有图片文件
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(folder_path).glob(f'*{ext}'))
                image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
            
            if not image_files:
                print(f"未在文件夹中找到图片文件")
                continue
            
            print(f"\n找到 {len(image_files)} 个图片文件")
            
            # 询问是否使用自定义阈值
            use_custom = input("是否使用自定义阈值？(y/n, 默认n): ").strip().lower()
            threshold = None
            if use_custom == 'y':
                try:
                    threshold = int(input("请输入阈值 (0-255, 推荐127): "))
                except ValueError:
                    print("无效输入，使用自动阈值")
            
            success_count = 0
            for img_path in image_files:
                signature = extractor.extract_from_file(str(img_path), threshold)
                if signature is not None:
                    success_count += 1
            
            print(f"\n批量处理完成: 成功提取 {success_count}/{len(image_files)} 个签名")
        
        else:
            print("无效选项，请重新输入")


if __name__ == "__main__":
    main()

