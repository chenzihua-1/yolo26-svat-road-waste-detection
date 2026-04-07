from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def visualize_predictions(model_path, image_path, save_path=None):
    """可视化单张图片预测结果"""
    # 加载模型
    model = YOLO(model_path)

    # 进行预测
    results = model(image_path)

    # 获取原始图片
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 绘制检测框和标签
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                # 绘制红色检测框（3像素粗）
                cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

                # 添加类别+置信度标签
                label = f'{model.names[cls]}: {conf:.2f}'
                cv2.putText(image_rgb, label, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 显示并保存图片
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('YOLO Detection Results')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"单张结果保存至: {save_path}")

    plt.show()


def batch_visualize_predictions(model_path, images_dir, output_dir):
    """批量处理图片并保存检测结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    model = YOLO(model_path)

    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))

    print(f"找到 {len(image_files)} 张图片，开始批量检测...")

    for i, image_path in enumerate(image_files):
        try:
            # 预测并绘制结果
            results = model(str(image_path))
            image = cv2.imread(str(image_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                        label = f'{model.names[cls]}: {conf:.2f}'
                        cv2.putText(image_rgb, label, (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 保存结果
            output_path = os.path.join(output_dir, f'detected_{image_path.name}')
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.title(f'Detection Results - {image_path.name}')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()

            print(f"已处理 ({i + 1}/{len(image_files)}): {image_path.name}")

        except Exception as e:
            print(f"处理 {image_path.name} 出错: {e}")

    print(f"\n批量检测完成！结果保存在: {output_dir}")


def plot_training_results(results_dir):
    """绘制训练结果图表（若存在）"""
    results_path = Path(results_dir) / 'results.png'
    if results_path.exists():
        img = cv2.imread(str(results_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(15, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title('Training Results')
        plt.show()
    else:
        print(f"未找到训练结果图: {results_path}")


if __name__ == '__main__':
    # 路径配置（替换为你的实际路径）
    model_path = r'C:\Users\56825\Desktop\yolo266\runs\detect\runs\train\exp9\weights\best.pt'
    images_dir = r'C:\Users\56825\Desktop\yolo266\datasets\images\val'
    output_dir = r'C:\Users\56825\Desktop\yolo266\jieguo'

    # 批量检测并保存结果
    batch_visualize_predictions(model_path, images_dir, output_dir)

    # （可选）绘制训练结果（若需查看训练曲线）
    # plot_training_results(r'C:\Users\56825\Desktop\yolo266\runs\train\exp3')

    print("检测任务已完成！")